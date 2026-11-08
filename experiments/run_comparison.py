"""Experiment 1: SPO vs GRPO head-to-head (analog of Table 1 / Section 3.1).

Trains both methods on the same contextual bandit under matched compute
(equal total environment interactions per iteration) and plots:
  * mean true value V_pi over training (convergence + final performance)
  * GRPO degenerate-group ratio vs SPO near-zero-advantage ratio (Figure 3a)

Run:  python experiments/run_comparison.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from spo import ContextualBandit, SPOTrainer, SPOConfig, GRPOTrainer, GRPOConfig

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run_seed(seed, n_iters=300, batch=256, group=8):
    env = ContextualBandit(n_states=50, n_actions=5, seed=seed)

    spo = SPOTrainer(env, SPOConfig(
        batch_size=batch, n_iterations=n_iters, lr=0.2, seed=seed))
    spo_m = spo.train()

    # Fresh identical env for GRPO (same seed -> same bandit).
    env2 = ContextualBandit(n_states=50, n_actions=5, seed=seed)
    grpo = GRPOTrainer(env2, GRPOConfig(
        total_responses=batch, group_size=group, n_iterations=n_iters, lr=0.2, seed=seed))
    grpo_m = grpo.train()

    return spo_m, grpo_m


def main(seeds=(0, 1, 2, 3, 4), n_iters=300):
    spo_vals, grpo_vals = [], []
    spo_nz, grpo_deg = [], []
    spo_mae = []

    for s in seeds:
        sm, gm = run_seed(s, n_iters=n_iters)
        spo_vals.append(sm.mean_value)
        grpo_vals.append(gm.mean_value)
        spo_nz.append(sm.near_zero_ratio_loose)
        grpo_deg.append(gm.degenerate_ratio)
        spo_mae.append(sm.tracker_mae)
        print(f"seed {s}: SPO final V={sm.mean_value[-1]:.4f}  "
              f"GRPO final V={gm.mean_value[-1]:.4f}  "
              f"(delta={sm.mean_value[-1]-gm.mean_value[-1]:+.4f})")

    spo_vals = np.array(spo_vals); grpo_vals = np.array(grpo_vals)
    iters = sm.iters

    spo_mean, spo_se = spo_vals.mean(0), spo_vals.std(0) / np.sqrt(len(seeds))
    grpo_mean, grpo_se = grpo_vals.mean(0), grpo_vals.std(0) / np.sqrt(len(seeds))

    print("\n=== FINAL (mean over seeds) ===")
    print(f"SPO  mean value: {spo_mean[-1]:.4f} +/- {spo_se[-1]:.4f}")
    print(f"GRPO mean value: {grpo_mean[-1]:.4f} +/- {grpo_se[-1]:.4f}")
    print(f"Improvement: {(spo_mean[-1]-grpo_mean[-1])*100:+.2f} percentage points")

    # ---------------- Figure: convergence + signal efficiency ---------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(iters, spo_mean, color="#d98a29", lw=2, label="SPO")
    ax.fill_between(iters, spo_mean - spo_se, spo_mean + spo_se, color="#d98a29", alpha=0.2)
    ax.plot(iters, grpo_mean, color="#2a7fd9", lw=2, label="GRPO")
    ax.fill_between(iters, grpo_mean - grpo_se, grpo_mean + grpo_se, color="#2a7fd9", alpha=0.2)
    ax.set_xlabel("Training Iteration"); ax.set_ylabel(r"Mean true value $V_\pi$")
    ax.set_title("Convergence: SPO vs GRPO (matched compute)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    grpo_deg = np.array(grpo_deg).mean(0)
    spo_nz = np.array(spo_nz).mean(0)
    ax.plot(iters, grpo_deg, color="#2a7fd9", lw=2, label="GRPO degenerate groups")
    ax.plot(iters, spo_nz, color="#d98a29", lw=2, label=r"SPO near-zero $|A|\leq0.02$")
    ax.set_xlabel("Training Iteration"); ax.set_ylabel("Ratio")
    ax.set_title("Signal efficiency (Figure 3a analog)")
    ax.set_ylim(-0.02, 1.0); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "comparison.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")
    return spo_mean, grpo_mean


if __name__ == "__main__":
    main()
