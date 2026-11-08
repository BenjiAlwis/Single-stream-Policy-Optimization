"""KL-Adaptive Bayesian Value Tracker (Section 2.1 of the paper).

This is the heart of SPO. For each "state" s (the general-RL analog of a
"prompt" x in the LLM paper) it maintains a *persistent*, low-variance estimate
of the policy's success probability V_pi(s) = E_{a~pi}[R(s, a)].

For binary rewards r in {0, 1} the success probability is modeled with a Beta
distribution (the conjugate prior for the Bernoulli outcome process):

    v_hat(s) ~ Beta(alpha(s), beta(s)),   v_hat(s) = alpha / (alpha + beta).

Crucially, the tracker stores only O(1) scalars per state (alpha, beta and a
snapshot of the last-acting policy's logits at that state). It never buffers a
history of past rewards: the recursive Beta update *summarizes* the entire
history in two numbers.

Update (Eq. 1), applied after observing reward r:

    alpha <- rho * alpha + r
    beta  <- rho * beta  + (1 - r)

where the discount rho in [rho_min, rho_max] is driven by the KL divergence
D(s) between the current policy and the last policy that acted on s (Eq. just
below Eq. 1):

    rho(s) = 2 ** (-D(s) / D_half)

so the tracker forgets faster when the policy has changed a lot since it last
saw s.

The Beta update is provably equivalent to an adaptive EMA on the value estimate
(Eq. 3):

    v_hat <- v_hat + eta * (r - v_hat),   eta = 1 / (rho * N_eff_prev + 1)

which is used directly for general (non-binary) rewards. We expose both views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class BetaState:
    """Per-state O(1) sufficient statistics maintained by the tracker."""

    alpha: float
    beta: float
    # Snapshot of the action-logits of the policy that *last acted* on this
    # state, used to compute the KL divergence that drives forgetting.
    # None until the state has been acted on at least once after init.
    last_logits: Optional[np.ndarray] = None

    @property
    def v_hat(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n_eff(self) -> float:
        # Effective sample count. The "+1" matches the paper's
        # N_eff = alpha + beta + 1 used in the EMA learning-rate (Eq. 3).
        return self.alpha + self.beta + 1.0


class KLAdaptiveTracker:
    """Persistent, KL-adaptive Beta/EMA value tracker over a set of states.

    Parameters
    ----------
    rho_min, rho_max : float
        Bounds on the forgetting factor. rho_min sets the *maximum* memory
        (largest window), rho_max sets the minimum forgetting. The paper uses
        rho_min = 0.875 (window W_min = 1/(1-rho_min) = 8) and rho_max = 0.96
        (W_max = 25).
    d_half : float
        KL value at which rho is halved relative to no-forgetting. Controls how
        aggressively the tracker discounts old evidence as the policy drifts.
    binary : bool
        If True (default) maintain Beta (alpha, beta). If False, track v_hat
        directly with the equivalent EMA form for general rewards.
    """

    def __init__(
        self,
        rho_min: float = 0.875,
        rho_max: float = 0.96,
        d_half: float = 0.5,
        binary: bool = True,
    ) -> None:
        assert 0.0 < rho_min <= rho_max < 1.0
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.d_half = d_half
        self.binary = binary

        # Initial effective sample size set to its expected equilibrium so the
        # tracker does not suffer transient instability (Section 2.1, Init).
        # N0 = 1 / (1 - rho_min).
        self.N0 = 1.0 / (1.0 - rho_min)

        self._states: Dict[int, BetaState] = {}
        # EMA-only view (used when binary=False): v_hat and effective count.
        self._ema_v: Dict[int, float] = {}
        self._ema_neff: Dict[int, float] = {}

    # ------------------------------------------------------------------ #
    # Initialization (Algorithm 2)
    # ------------------------------------------------------------------ #
    def init_state(self, sid: int, v0: float) -> None:
        """Warm-start a state from an offline value estimate v0 (Eq. 2).

            alpha0 = N0 * v0,   beta0 = N0 * (1 - v0).
        """
        v0 = float(np.clip(v0, 1e-6, 1.0 - 1e-6))
        if self.binary:
            self._states[sid] = BetaState(alpha=self.N0 * v0, beta=self.N0 * (1.0 - v0))
        else:
            self._ema_v[sid] = v0
            self._ema_neff[sid] = self.N0

    def has(self, sid: int) -> bool:
        return sid in self._states or sid in self._ema_v

    # ------------------------------------------------------------------ #
    # Forgetting factor from KL divergence
    # ------------------------------------------------------------------ #
    def rho_from_kl(self, kl: float) -> float:
        """rho(s) = 2 ** (-D(s) / D_half), clamped to [rho_min, rho_max]."""
        rho = 2.0 ** (-max(kl, 0.0) / self.d_half)
        return float(np.clip(rho, self.rho_min, self.rho_max))

    # ------------------------------------------------------------------ #
    # Value query (the baseline used for advantage estimation)
    # ------------------------------------------------------------------ #
    def v_hat(self, sid: int) -> float:
        if self.binary:
            return self._states[sid].v_hat
        return self._ema_v[sid]

    def n_eff(self, sid: int) -> float:
        if self.binary:
            return self._states[sid].n_eff
        return self._ema_neff[sid] + 1.0

    def last_logits(self, sid: int) -> Optional[np.ndarray]:
        if self.binary and sid in self._states:
            return self._states[sid].last_logits
        return None

    # ------------------------------------------------------------------ #
    # The core update (Eq. 1 / Eq. 3)
    # ------------------------------------------------------------------ #
    def update(
        self,
        sid: int,
        reward: float,
        kl: float = 0.0,
        cur_logits: Optional[np.ndarray] = None,
    ) -> float:
        """Fold a new observation into the tracker. Returns the new v_hat.

        Parameters
        ----------
        sid : int
            State identifier.
        reward : float
            Observed reward (in {0, 1} when binary=True; any value otherwise).
        kl : float
            KL divergence D(s) between the current policy and the policy that
            last acted on s, used to set the forgetting factor.
        cur_logits : np.ndarray, optional
            The current policy's action logits at s, snapshotted so the *next*
            update can measure how far the policy has drifted.
        """
        rho = self.rho_from_kl(kl)

        if self.binary:
            st = self._states[sid]
            st.alpha = rho * st.alpha + reward
            st.beta = rho * st.beta + (1.0 - reward)
            if cur_logits is not None:
                st.last_logits = np.array(cur_logits, dtype=np.float64, copy=True)
            return st.v_hat

        # EMA form for general rewards (Eq. 3): equivalent to the Beta update.
        n_prev = self._ema_neff[sid]
        eta = 1.0 / (rho * n_prev + 1.0)
        v_prev = self._ema_v[sid]
        v_new = v_prev + eta * (reward - v_prev)
        self._ema_v[sid] = v_new
        # Effective count evolves as N <- rho*N + 1 (matches the Beta mass).
        self._ema_neff[sid] = rho * n_prev + 1.0
        return v_new

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def all_v_hat(self) -> np.ndarray:
        if self.binary:
            return np.array([self._states[s].v_hat for s in sorted(self._states)])
        return np.array([self._ema_v[s] for s in sorted(self._ema_v)])

    def state_ids(self):
        return sorted(self._states) if self.binary else sorted(self._ema_v)
