"""Prioritized sampling (Section 2.3, Eq. 7) and advantage estimation (Eqs. 4-5).

Prioritized sampling
--------------------
SPO samples states with weight proportional to the estimated Bernoulli standard
deviation plus an exploration bonus (Eq. 7):

    w(s) proportional to sqrt( v_hat(s) * (1 - v_hat(s)) ) + epsilon

This is an adaptive curriculum: it up-weights states of medium difficulty
(v_hat ~ 0.5, highest learning potential) and down-weights states that are
already mastered (v_hat ~ 1) or hopeless (v_hat ~ 0). The bonus epsilon = 0.05
keeps every state reachable, preventing curriculum collapse.

Advantage estimation
---------------------
Raw advantage uses the *pre-update* baseline so it is independent of the action
taken this step, keeping the policy gradient unbiased (Eq. 4):

    A(s, a) = r(s, a) - v_hat_prev(s)

Advantages are then normalized *globally across the batch* (Eq. 5), not
per-group:

    A_tilde = (A - mu_B) / sigma_B
"""

from __future__ import annotations

import numpy as np


def sampling_weights(v_hat: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """Eq. 7: w(s) ∝ sqrt(v_hat (1 - v_hat)) + eps."""
    std = np.sqrt(np.clip(v_hat * (1.0 - v_hat), 0.0, None))
    w = std + eps
    return w / w.sum()


def sample_states(
    v_hat: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    eps: float = 0.05,
    replace: bool = False,
) -> np.ndarray:
    """Draw a batch of state indices according to the prioritized weights."""
    w = sampling_weights(v_hat, eps=eps)
    n = len(v_hat)
    if not replace and batch_size > n:
        replace = True
    return rng.choice(n, size=batch_size, replace=replace, p=w)


def raw_advantage(reward: float, v_hat_prev: float) -> float:
    """Eq. 4: A = r - v_hat_prev (pre-update baseline -> unbiased)."""
    return reward - v_hat_prev


def normalize_advantages(advs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Eq. 5: global batch normalization (A - mu_B) / sigma_B."""
    mu = advs.mean()
    sigma = advs.std()
    return (advs - mu) / (sigma + eps)
