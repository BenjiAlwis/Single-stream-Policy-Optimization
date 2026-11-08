"""Contextual bandit environment (the general-RL analog of the RLVR prompt set).

Mapping to the paper
--------------------
    prompt x          ->  context / state s   (one of `n_states` discrete states)
    response y         ->  action a            (one of `n_actions` discrete actions)
    binary reward r    ->  Bernoulli(success)  (r in {0, 1})

Each state s has a hidden optimal action a*(s). Taking the optimal action yields
success with probability `p_high`; any other action with probability `p_low`.
This gives every state a well-defined, learnable success probability under any
policy, exactly like a prompt's solve-rate under an evolving LLM policy.

States are assigned a spread of intrinsic difficulties so the curriculum and
degeneracy analyses have something to bite on: some states are easy (the gap
between p_high and p_low is large) and some are hard (small gap), mirroring the
easy/hard prompts that drive GRPO's degeneracy.
"""

from __future__ import annotations

import numpy as np


class ContextualBandit:
    def __init__(
        self,
        n_states: int = 200,
        n_actions: int = 10,
        p_high: float = 0.9,
        p_low: float = 0.05,
        difficulty_spread: bool = True,
        seed: int = 0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        # Hidden optimal action per state.
        self.optimal_action = self.rng.integers(0, n_actions, size=n_states)

        if difficulty_spread:
            # Give each state its own (p_low_s, p_high_s) so success
            # probabilities under the optimal policy span the full [low, high]
            # range -- creating genuinely easy and hard states.
            frac = self.rng.uniform(0.0, 1.0, size=n_states)
            self.p_high_s = p_low + (p_high - p_low) * frac  # optimal-action success
            self.p_low_s = np.full(n_states, p_low)
        else:
            self.p_high_s = np.full(n_states, p_high)
            self.p_low_s = np.full(n_states, p_low)

    def success_prob(self, state: int, action: int) -> float:
        """P(reward = 1 | state, action)."""
        if action == self.optimal_action[state]:
            return float(self.p_high_s[state])
        return float(self.p_low_s[state])

    def step(self, state: int, action: int) -> int:
        """Sample a binary reward for taking `action` in `state`."""
        p = self.success_prob(state, action)
        return int(self.rng.random() < p)

    # ------- ground-truth values, for evaluation / sanity checks -------- #
    def true_value(self, state: int, action_probs: np.ndarray) -> float:
        """True V_pi(s) = sum_a pi(a|s) * P(success | s, a)."""
        a_star = self.optimal_action[state]
        p = np.full(self.n_actions, self.p_low_s[state])
        p[a_star] = self.p_high_s[state]
        return float(np.dot(action_probs, p))

    def optimal_value(self, state: int) -> float:
        """Best achievable success probability (always pick a*)."""
        return float(self.p_high_s[state])
