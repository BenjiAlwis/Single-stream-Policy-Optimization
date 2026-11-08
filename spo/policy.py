"""Tabular softmax policy with a PPO-Clip update (Section 2.2, Eq. 6).

The paper notes "the development of core RL algorithms was on tabular
representation" (footnote 3), so a tabular policy is the most faithful
parameterization for this reduction. Logits theta[s, a] define

    pi(a | s) = softmax_a(theta[s, :]).

We implement the clipped surrogate objective (Eq. 6) with the Clip-Higher
trick from DAPO (eps_low, eps_high) used in the paper's setup, and provide the
per-state KL divergence used to drive the tracker's adaptive forgetting.
"""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class TabularPolicy:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        eps_low: float = 0.2,
        eps_high: float = 0.28,  # Clip-Higher (DAPO), as in the paper
        seed: int = 0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.eps_low = eps_low
        self.eps_high = eps_high
        rng = np.random.default_rng(seed)
        # Small random init so the policy is not perfectly uniform.
        self.theta = 0.01 * rng.standard_normal((n_states, n_actions))

    # ------------------------------------------------------------------ #
    def probs(self, state: int) -> np.ndarray:
        return _softmax(self.theta[state])

    def all_probs(self) -> np.ndarray:
        return _softmax(self.theta)

    def sample(self, state: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_actions, p=self.probs(state)))

    def logits(self, state: int) -> np.ndarray:
        return self.theta[state].copy()

    def log_prob(self, state: int, action: int) -> float:
        return float(np.log(self.probs(state)[action] + 1e-12))

    # ------------------------------------------------------------------ #
    @staticmethod
    def kl_from_logits(old_logits: np.ndarray, new_logits: np.ndarray) -> float:
        """KL( pi_old(.|s) || pi_new(.|s) ) at a single state.

        Used as D(s) to drive the tracker's forgetting factor rho(s).
        """
        p_old = _softmax(old_logits)
        p_new = _softmax(new_logits)
        return float(np.sum(p_old * (np.log(p_old + 1e-12) - np.log(p_new + 1e-12))))

    # ------------------------------------------------------------------ #
    def ppo_clip_update(self, batch, n_epochs: int = 1, minibatch: int | None = None):
        """One PPO-Clip optimization phase over a collected batch.

        Parameters
        ----------
        batch : list of (state, action, advantage, old_log_prob)
            One single-stream sample per element (SPO collects exactly one
            (state, action) per state per iteration).
        n_epochs : int
            Number of passes over the batch (mini-batch updates, footnote 4).
        minibatch : int, optional
            Mini-batch size; defaults to the full batch.
        """
        n = len(batch)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        advs = np.array([b[2] for b in batch], dtype=np.float64)
        old_logps = np.array([b[3] for b in batch], dtype=np.float64)

        mb = minibatch or n
        idx_all = np.arange(n)

        for _ in range(n_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, mb):
                idx = idx_all[start : start + mb]
                self._apply_clip_grad(states[idx], actions[idx], advs[idx], old_logps[idx])

    def _apply_clip_grad(self, states, actions, advs, old_logps):
        """Gradient of the PPO-Clip surrogate w.r.t. tabular logits.

        L_CLIP = E[ min(ratio * A, clip(ratio, 1-eps_low, 1+eps_high) * A) ].
        We ascend L_CLIP. Where the clip is active the surrogate is flat, so the
        gradient contribution is zeroed for those samples.
        """
        grad = np.zeros_like(self.theta)

        for s, a, adv, old_lp in zip(states, actions, advs, old_logps):
            p = self.probs(s)
            cur_lp = np.log(p[a] + 1e-12)
            ratio = np.exp(cur_lp - old_lp)

            lo, hi = 1.0 - self.eps_low, 1.0 + self.eps_high
            # Determine whether the clipped branch is the active (min) one.
            unclipped = ratio * adv
            clipped = np.clip(ratio, lo, hi) * adv
            use_clipped = clipped < unclipped

            # If the clipped branch is active, the objective is flat in ratio
            # -> no gradient. Otherwise grad of ratio*A w.r.t logits.
            if use_clipped and (
                (ratio <= lo and adv < 0) or (ratio >= hi and adv > 0)
                or (ratio < lo or ratio > hi)
            ):
                # Surrogate saturated; skip (zero gradient).
                continue

            # d/dtheta[s, j] of log pi(a|s) = (1[j==a] - pi(j|s)).
            dlogp = -p.copy()
            dlogp[a] += 1.0
            # d ratio / d theta = ratio * dlogp ; d(ratio*A) = A * ratio * dlogp.
            grad[s] += adv * ratio * dlogp

        grad /= max(len(states), 1)
        self.theta += self.lr * grad
