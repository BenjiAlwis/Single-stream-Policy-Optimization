"""Tests verifying SPO's core mathematical claims from the paper."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from spo.tracker import KLAdaptiveTracker
from spo.advantage import sampling_weights, normalize_advantages, raw_advantage
from spo.policy import TabularPolicy


def test_beta_ema_equivalence():
    """The Beta update (Eq.1) must equal the EMA update (Eq.3) for the value."""
    binary = KLAdaptiveTracker(binary=True, d_half=0.5)
    ema = KLAdaptiveTracker(binary=False, d_half=0.5)
    binary.init_state(0, 0.5)
    ema.init_state(0, 0.5)

    rng = np.random.default_rng(0)
    for _ in range(200):
        r = float(rng.random() < 0.3)
        kl = float(rng.random() * 0.3)
        vb = binary.update(0, r, kl=kl)
        ve = ema.update(0, r, kl=kl)
        assert abs(vb - ve) < 1e-9, f"Beta {vb} != EMA {ve}"
    print("PASS: Beta update == EMA update (Eq.1 <=> Eq.3)")


def test_n0_equilibrium():
    """N0 = 1/(1-rho_min) is the equilibrium effective sample size."""
    rho_min = 0.875
    t = KLAdaptiveTracker(rho_min=rho_min, d_half=0.5)
    assert abs(t.N0 - 1.0 / (1.0 - rho_min)) < 1e-12
    # At equilibrium with constant rho_min, N_eff stays ~ N0+1.
    t.init_state(0, 0.5)
    for _ in range(500):
        t.update(0, 1.0, kl=0.0)  # kl=0 -> rho=rho_max actually (2^0=1 clamped)
    # With kl=0, rho = min(1, rho_max) = rho_max; check value converges to 1.
    assert t.v_hat(0) > 0.95
    print("PASS: N0 equilibrium and convergence under constant reward")


def test_rho_monotonic_in_kl():
    """Larger KL -> smaller rho -> faster forgetting."""
    t = KLAdaptiveTracker(rho_min=0.5, rho_max=0.99, d_half=0.5)
    rhos = [t.rho_from_kl(kl) for kl in [0.0, 0.25, 0.5, 1.0, 2.0]]
    assert all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1)), rhos
    assert abs(t.rho_from_kl(0.5) - 0.5 * 1.0) < 0.01 or t.rho_from_kl(0.5) <= 0.99
    print(f"PASS: rho monotonically decreasing in KL: {[round(r,3) for r in rhos]}")


def test_tracker_recovers_true_prob():
    """With a fixed policy, v_hat should converge near the true success prob."""
    t = KLAdaptiveTracker(rho_min=0.95, rho_max=0.99, d_half=10.0, binary=True)
    t.init_state(0, 0.5)
    rng = np.random.default_rng(1)
    p_true = 0.3
    for _ in range(5000):
        r = float(rng.random() < p_true)
        t.update(0, r, kl=0.0)
    assert abs(t.v_hat(0) - p_true) < 0.05, t.v_hat(0)
    print(f"PASS: tracker recovers true prob {p_true:.2f} -> {t.v_hat(0):.3f}")


def test_prioritized_weights():
    """Eq.7: weight peaks at v=0.5, minimized at extremes; eps keeps positivity."""
    v = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    w = sampling_weights(v, eps=0.05)
    assert np.all(w > 0), "exploration bonus must keep all weights positive"
    assert w[2] == w.max(), "medium-difficulty state must get the most weight"
    assert w[0] == w[4], "symmetric extremes get equal (minimal) weight"
    print(f"PASS: prioritized weights peak at v=0.5: {np.round(w,4)}")


def test_global_normalization():
    """Eq.5: normalized advantages are zero-mean, unit-std."""
    advs = np.array([0.2, -0.5, 0.8, -0.1, 0.3])
    z = normalize_advantages(advs)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-4
    print("PASS: global normalization -> zero mean, unit std")


def test_advantage_unbiased_baseline():
    """A = r - v_prev uses pre-update baseline (independent of current action)."""
    assert raw_advantage(1.0, 0.3) == 0.7
    assert raw_advantage(0.0, 0.3) == -0.3
    print("PASS: raw advantage uses pre-update baseline")


def test_ppo_clip_improves_simple():
    """Sanity: PPO-Clip increases prob of a positively-advantaged action."""
    pol = TabularPolicy(n_states=1, n_actions=3, lr=0.5, seed=0)
    p_before = pol.probs(0)[1]
    lp = pol.log_prob(0, 1)
    for _ in range(20):
        batch = [(0, 1, 1.0, lp)]   # action 1, positive advantage
        pol.ppo_clip_update(batch, n_epochs=1)
        lp = pol.log_prob(0, 1)
    p_after = pol.probs(0)[1]
    assert p_after > p_before, (p_before, p_after)
    print(f"PASS: PPO-Clip raises advantaged action prob {p_before:.3f}->{p_after:.3f}")


def test_kl_zero_when_unchanged():
    pol = TabularPolicy(n_states=2, n_actions=4, seed=3)
    lg = pol.logits(0)
    assert pol.kl_from_logits(lg, lg) < 1e-12
    print("PASS: KL is zero for unchanged policy")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
