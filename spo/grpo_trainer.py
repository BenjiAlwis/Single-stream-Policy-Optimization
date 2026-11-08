"""GRPO baseline (group-based) for head-to-head comparison.

Mirrors the SPO trainer but uses the group-based machinery the paper critiques:
  * generates a group of G actions per sampled state            (Eq. 11)
  * baseline = group mean reward; normalize within the group    (per-group)
  * degenerate groups (all-0 or all-1) yield zero advantage -> no signal

Uniform prompt sampling (GRPO's standard formulation). For a *fair* compute
comparison with SPO we match total environment interactions per iteration:
SPO uses `batch_size` single streams; GRPO uses `batch_size / G` states each
expanded into G responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .envs import ContextualBandit
from .policy import TabularPolicy


@dataclass
class GRPOConfig:
    total_responses: int = 256       # matched to SPO batch_size for fair compute
    group_size: int = 8
    n_iterations: int = 150
    ppo_epochs: int = 4
    minibatch: int | None = None
    lr: float = 0.1
    eps_low: float = 0.2
    eps_high: float = 0.28
    seed: int = 0


@dataclass
class GRPOMetrics:
    iters: List[int] = field(default_factory=list)
    mean_value: List[float] = field(default_factory=list)
    success_rate: List[float] = field(default_factory=list)
    degenerate_ratio: List[float] = field(default_factory=list)     # fraction of groups all-same
    effective_adv_variance: List[float] = field(default_factory=list)  # var over non-degenerate samples


class GRPOTrainer:
    def __init__(self, env: ContextualBandit, config: GRPOConfig):
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
        self.metrics = GRPOMetrics()

    def _true_mean_value(self) -> float:
        probs = self.policy.all_probs()
        return float(
            np.mean([self.env.true_value(s, probs[s]) for s in range(self.env.n_states)])
        )

    def train(self, log_every: int = 1):
        cfg = self.cfg
        G = cfg.group_size
        n_groups = max(cfg.total_responses // G, 1)

        for i in range(cfg.n_iterations):
            batch = []          # (state, action, advantage, old_log_prob)
            rewards_all = []
            degenerate = 0
            eff_advs = []

            # uniform sampling of prompts/states
            group_states = self.rng.integers(0, self.env.n_states, size=n_groups)

            for s in group_states:
                s = int(s)
                actions = [self.policy.sample(s, self.rng) for _ in range(G)]
                rewards = np.array([self.env.step(s, a) for a in actions], dtype=float)
                old_lps = [self.policy.log_prob(s, a) for a in actions]
                rewards_all.extend(rewards.tolist())

                mu = rewards.mean()
                sigma = rewards.std()

                # Degenerate group: all rewards identical -> zero signal.
                if sigma < 1e-12:
                    degenerate += 1
                    advs = np.zeros(G)              # no learning signal
                else:
                    advs = (rewards - mu) / (sigma + 1e-8)   # per-group normalization
                    eff_advs.extend(advs.tolist())

                for a, adv, lp in zip(actions, advs, old_lps):
                    batch.append((s, a, float(adv), lp))

            self.policy.ppo_clip_update(
                batch, n_epochs=cfg.ppo_epochs, minibatch=cfg.minibatch
            )

            if i % log_every == 0:
                self.metrics.iters.append(i)
                self.metrics.mean_value.append(self._true_mean_value())
                self.metrics.success_rate.append(float(np.mean(rewards_all)))
                self.metrics.degenerate_ratio.append(degenerate / n_groups)
                self.metrics.effective_adv_variance.append(
                    float(np.var(eff_advs)) if eff_advs else 0.0
                )

        return self.metrics
