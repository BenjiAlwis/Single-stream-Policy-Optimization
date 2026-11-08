"""Single-stream Policy Optimization trainer (Algorithm 1 + Algorithm 2 init).

Ties together the tracker, prioritized sampler, advantage pipeline and PPO-Clip
policy into the full SPO loop, generalized from prompts to bandit contexts.

Algorithm 1 (per iteration i):
    1. compute sampling weights w_i(s)           (Eq. 7)
    2. sample a batch B of states by weight
    3. for each state s in B:
         a. sample action a ~ pi_{i-1}(.|s), observe reward r            (single stream!)
         b. raw advantage A = r - v_hat_{i-1}(s)                         (Eq. 4)
         c. update tracker v_hat(s) with KL-adaptive forgetting          (Eq. 1)
    4. normalize advantages across the batch                            (Eq. 5)
    5. update policy with PPO-Clip                                       (Eq. 6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .advantage import (
    normalize_advantages,
    raw_advantage,
    sample_states,
    sampling_weights,
)
from .envs import ContextualBandit
from .policy import TabularPolicy
from .tracker import KLAdaptiveTracker


@dataclass
class SPOConfig:
    batch_size: int = 256
    n_iterations: int = 150
    ppo_epochs: int = 4
    minibatch: int | None = None
    # Tracker
    rho_min: float = 0.875
    rho_max: float = 0.96
    d_half: float = 0.5
    # Sampler
    explore_eps: float = 0.05
    prioritized: bool = True
    no_batchnorm: bool = False   # ablation: use raw (unnormalized) advantages
    # Offline init (Algorithm 2)
    n0: int = 8
    offline_init: bool = True
    # Policy
    lr: float = 0.1
    eps_low: float = 0.2
    eps_high: float = 0.28
    seed: int = 0


@dataclass
class SPOMetrics:
    iters: List[int] = field(default_factory=list)
    mean_value: List[float] = field(default_factory=list)        # true V_pi averaged over states
    success_rate: List[float] = field(default_factory=list)      # realized batch success
    adv_variance: List[float] = field(default_factory=list)      # var of raw advantages
    near_zero_ratio_tight: List[float] = field(default_factory=list)   # |A| <= 1e-4
    near_zero_ratio_loose: List[float] = field(default_factory=list)   # |A| <= 0.02
    tracker_mae: List[float] = field(default_factory=list)       # |v_hat - true V| mean


class SPOTrainer:
    def __init__(self, env: ContextualBandit, config: SPOConfig):
        self.env = env
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.policy = TabularPolicy(
            env.n_states,
            env.n_actions,
            lr=config.lr,
            eps_low=config.eps_low,
            eps_high=config.eps_high,
            seed=config.seed,
        )
        self.tracker = KLAdaptiveTracker(
            rho_min=config.rho_min,
            rho_max=config.rho_max,
            d_half=config.d_half,
            binary=True,
        )
        self.metrics = SPOMetrics()
        self._init_tracker()

    # ------------------------------------------------------------------ #
    def _init_tracker(self):
        """Algorithm 2: warm-start the tracker from n0 offline samples."""
        for s in range(self.env.n_states):
            if self.cfg.offline_init:
                outcomes = [
                    self.env.step(s, self.policy.sample(s, self.rng))
                    for _ in range(self.cfg.n0)
                ]
                v0 = float(np.mean(outcomes))
            else:
                v0 = 0.5  # uninformed prior
            self.tracker.init_state(s, v0)
            # Snapshot the initializing policy's logits for the first KL.
            self.tracker._states[s].last_logits = self.policy.logits(s)

    # ------------------------------------------------------------------ #
    def _true_mean_value(self) -> float:
        probs = self.policy.all_probs()
        return float(
            np.mean([self.env.true_value(s, probs[s]) for s in range(self.env.n_states)])
        )

    def _tracker_mae(self) -> float:
        probs = self.policy.all_probs()
        errs = [
            abs(self.tracker.v_hat(s) - self.env.true_value(s, probs[s]))
            for s in range(self.env.n_states)
        ]
        return float(np.mean(errs))

    # ------------------------------------------------------------------ #
    def train(self, log_every: int = 1):
        cfg = self.cfg
        for i in range(cfg.n_iterations):
            # (1-2) prioritized sampling of a state batch
            v_all = np.array([self.tracker.v_hat(s) for s in range(self.env.n_states)])
            if cfg.prioritized:
                batch_states = sample_states(
                    v_all, cfg.batch_size, self.rng, eps=cfg.explore_eps
                )
            else:
                batch_states = self.rng.integers(0, self.env.n_states, size=cfg.batch_size)

            samples = []          # (state, action, old_log_prob)
            raw_advs = []
            rewards = []

            # (3) single stream per state: one action, one reward
            for s in batch_states:
                s = int(s)
                a = self.policy.sample(s, self.rng)
                r = self.env.step(s, a)
                old_lp = self.policy.log_prob(s, a)

                v_prev = self.tracker.v_hat(s)        # pre-update baseline (Eq. 4)
                A = raw_advantage(r, v_prev)

                # KL between current policy and last-acting policy at s.
                last_logits = self.tracker.last_logits(s)
                cur_logits = self.policy.logits(s)
                kl = (
                    self.policy.kl_from_logits(last_logits, cur_logits)
                    if last_logits is not None
                    else 0.0
                )
                # (3c) update tracker, snapshot current logits for next time.
                self.tracker.update(s, r, kl=kl, cur_logits=cur_logits)

                samples.append((s, a, old_lp))
                raw_advs.append(A)
                rewards.append(r)

            raw_advs = np.array(raw_advs)

            # (4) global batch normalization (Eq. 5)
            if cfg.no_batchnorm:
                norm_advs = raw_advs            # ablation: skip normalization
            else:
                norm_advs = normalize_advantages(raw_advs)

            # (5) PPO-Clip update
            batch = [
                (s, a, float(adv), old_lp)
                for (s, a, old_lp), adv in zip(samples, norm_advs)
            ]
            self.policy.ppo_clip_update(
                batch, n_epochs=cfg.ppo_epochs, minibatch=cfg.minibatch
            )

            # ---- metrics ----
            if i % log_every == 0:
                self.metrics.iters.append(i)
                self.metrics.mean_value.append(self._true_mean_value())
                self.metrics.success_rate.append(float(np.mean(rewards)))
                self.metrics.adv_variance.append(float(np.var(raw_advs)))
                self.metrics.near_zero_ratio_tight.append(
                    float(np.mean(np.abs(raw_advs) <= 1e-4))
                )
                self.metrics.near_zero_ratio_loose.append(
                    float(np.mean(np.abs(raw_advs) <= 0.02))
                )
                self.metrics.tracker_mae.append(self._tracker_mae())

        return self.metrics
