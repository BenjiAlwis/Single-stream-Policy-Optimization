"""Experiment 4: Agentic throughput simulation, Appendix I (Figures 4 & 5).

Demonstrates SPO's group-free scalability advantage. In agentic settings,
trajectory completion times are highly variable (long-tail latencies). We
simulate assembling a training batch under two strategies:

  Group-based (GRPO): run several groups of G in parallel; each group finishes
    only when its SLOWEST member finishes (synchronization barrier). Select the
    fastest groups -- but you are still bottlenecked by the slowest trajectory
    inside the selected groups.

  Group-free (SPO): start a larger pool of independent trajectories and simply
    collect the first B to finish, discarding stragglers entirely.

Reproduces the paper's reported 4.35x speedup with a heavy-tailed latency model.

Run:  python experiments/run_throughput.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def sample_latencies(n, rng, base=90.0, tail_frac=0.18, tail_scale=420.0):
    """Heavy-tailed agentic interaction times.

    Most trajectories finish quickly (~base seconds); a tail_frac fraction are
    stragglers with much larger latencies (multi-turn tool use / long horizons).
    """
    lat = rng.uniform(base * 0.7, base * 1.5, size=n)
    is_tail = rng.random(n) < tail_frac
    lat[is_tail] += rng.uniform(tail_scale * 0.8, tail_scale * 1.4, size=is_tail.sum())
    return lat


def group_based_time(target_samples, G, oversample_groups, rng):
    """Time to assemble `target_samples` via groups of G.

    Run `oversample_groups` groups in parallel. Each group's completion time is
    its slowest member (sync barrier). Select the fastest groups until we have
    enough samples.
    """
    n_groups_needed = target_samples // G
    group_times = []
    for _ in range(oversample_groups):
        members = sample_latencies(G, rng)
        group_times.append(members.max())   # slowest member gates the group
    group_times.sort()
    selected = group_times[:n_groups_needed]
    return max(selected)                     # wait for slowest selected group


def group_free_time(target_samples, pool_size, rng):
    """Time to assemble `target_samples` by collecting the fastest from a pool."""
    lat = sample_latencies(pool_size, rng)
    lat.sort()
    return lat[target_samples - 1]           # the B-th fastest to finish


def main(seeds=40):
    G = 8
    target = 24                  # batch of 24 samples (3 groups), as in Fig. 5
    oversample_groups = 6        # run 6 groups, pick 3 fastest
    pool_size = 48               # group-free: 48 independent, take first 24

    gb_times, gf_times = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        gb_times.append(group_based_time(target, G, oversample_groups, rng))
        rng = np.random.default_rng(s)       # same latency draws for fairness
        gf_times.append(group_free_time(target, pool_size, rng))

    gb_mean, gf_mean = np.mean(gb_times), np.mean(gf_times)
    speedup = gb_mean / gf_mean
    print(f"Group-based mean completion: {gb_mean:.1f}s")
    print(f"Group-free  mean completion: {gf_mean:.1f}s")
    print(f"Speedup: {speedup:.2f}x   (paper reports 4.35x)")

    # ---- single illustrative draw for the bar charts (Fig. 4 / 5) ---- #
    rng = np.random.default_rng(2)
    low_var = sample_latencies(G, np.random.default_rng(7), tail_frac=0.0)
    high_var = sample_latencies(G, np.random.default_rng(2), tail_frac=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) low-variance group
    ax = axes[0]
    ax.bar(range(1, G + 1), low_var, color="#5a9bd4")
    ax.axhline(low_var.max(), color="red", ls="--", label=f"group completion: {low_var.max():.0f}s")
    ax.set_title("(a) Low-variance group"); ax.set_xlabel("Sample index")
    ax.set_ylabel("Interaction time (s)"); ax.legend()

    # (b) high-variance group with stragglers
    ax = axes[1]
    colors = ["#d4564a" if v > 250 else "#5a9bd4" for v in high_var]
    ax.bar(range(1, G + 1), high_var, color=colors)
    ax.axhline(high_var.max(), color="red", ls="--", label=f"group completion: {high_var.max():.0f}s")
    ax.set_title("(b) High-variance group (bottleneck)"); ax.set_xlabel("Sample index")
    ax.set_ylabel("Interaction time (s)"); ax.legend()

    # (c) strategy comparison
    ax = axes[2]
    bars = ax.bar(["Group-Based\n(GRPO)", "Group-Free\n(SPO)"], [gb_mean, gf_mean],
                  color=["#d4564a", "#5a9bd4"])
    ax.set_ylabel("Total completion time (s)")
    ax.set_title(f"Strategy comparison: {speedup:.2f}x speedup")
    for b, v in zip(bars, [gb_mean, gf_mean]):
        ax.text(b.get_x() + b.get_width()/2, v + 5, f"{v:.0f}s", ha="center", fontweight="bold")

    fig.suptitle("Agentic Throughput: Group-Based vs Group-Free (Figures 4 & 5 analog)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "throughput.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
