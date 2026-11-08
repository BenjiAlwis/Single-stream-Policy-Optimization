"""Experiment 2: Ablation studies (analog of Figure 2 and Appendix G).

Ablates SPO's core components, each averaged over seeds:
  (b) Baseline ablation       -- remove the value tracker (v_hat = 0)
  (c) Offline-init ablation    -- no warm start (uninformed v0 = 0.5)
  (d) Batch-norm ablation      -- use raw (unnormalized) advantages
  (e) Prioritized sampling     -- uniform vs prioritized state sampling

Run:  python experiments/run_ablations.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from spo import ContextualBandit, SPOTrainer, SPOConfig

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def train_variant(seed, n_iters, **overrides):
    env = ContextualBandit(n_states=50, n_actions=5, seed=seed)
    no_baseline = overrides.pop("_no_baseline", False)
    if overrides.pop("_no_batchnorm", False):
        overrides["no_batchnorm"] = True
    base = dict(batch_size=256, n_iterations=n_iters, lr=0.2, seed=seed)
    base.update(overrides)
    trainer = SPOTrainer(env, SPOConfig(**base))

    if no_baseline:
        # Force baseline 0 so advantage = raw reward (vanilla policy gradient).
        trainer.tracker.v_hat = lambda sid: 0.0  # type: ignore
        return trainer.train()
    return trainer.train()


def avg_over_seeds(seeds, n_iters, **overrides):
    runs = []
    for s in seeds:
        m = train_variant(s, n_iters, **dict(overrides))
        runs.append(m.mean_value)
    arr = np.array(runs)
    return arr.mean(0), arr.std(0) / np.sqrt(len(seeds))


def main(seeds=(0, 1, 2), n_iters=300):
    iters = list(range(n_iters))

    print("Running ablations (this trains several SPO variants)...")

    full_m, full_se = avg_over_seeds(seeds, n_iters)
    print(f"  full SPO            final V = {full_m[-1]:.4f}")

    nobase_m, nobase_se = avg_over_seeds(seeds, n_iters, _no_baseline=True)
    print(f"  w/o baseline        final V = {nobase_m[-1]:.4f}")

    noinit_m, noinit_se = avg_over_seeds(seeds, n_iters, offline_init=False)
    print(f"  w/o offline init    final V = {noinit_m[-1]:.4f}")

    # Batch-norm ablation: we approximate "no normalization" by a dedicated flag.
    # SPOConfig has no such flag, so emulate by very large batch std neutralization
    # is not trivial; instead we add a trainer-level option below.
    nonorm_m, nonorm_se = avg_over_seeds(seeds, n_iters, _no_batchnorm=True)
    print(f"  w/o batch norm      final V = {nonorm_m[-1]:.4f}")

    uniform_m, uniform_se = avg_over_seeds(seeds, n_iters, prioritized=False)
    print(f"  uniform sampling    final V = {uniform_m[-1]:.4f}")

    # ---------------- plot ---------------- #
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def panel(ax, title, m2, se2, label2, color2):
        ax.plot(iters, full_m, color="#2a7fd9", lw=2, label="full SPO")
        ax.fill_between(iters, full_m - full_se, full_m + full_se, color="#2a7fd9", alpha=0.2)
        ax.plot(iters, m2, color=color2, lw=2, label=label2)
        ax.fill_between(iters, m2 - se2, m2 + se2, color=color2, alpha=0.2)
        ax.set_title(title); ax.set_xlabel("Training Iteration")
        ax.set_ylabel(r"Mean true value $V_\pi$"); ax.legend(); ax.grid(alpha=0.3)

    panel(axes[0, 0], "(b) Baseline ablation", nobase_m, nobase_se, "w/o baseline", "#3aa64a")
    panel(axes[0, 1], "(c) Offline-init ablation", noinit_m, noinit_se, "w/o offline init", "#3aa64a")
    panel(axes[1, 0], "(d) Batch-norm ablation", nonorm_m, nonorm_se, "w/o batch norm", "#3aa64a")
    panel(axes[1, 1], "Prioritized vs uniform sampling", uniform_m, uniform_se, "uniform sampling", "#3aa64a")

    fig.suptitle("SPO Ablation Studies (Figure 2 analog)", fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ablations.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
