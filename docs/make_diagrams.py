"""Generate schematic diagrams for the SPO explainer PDF (matplotlib-based)."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
os.makedirs(OUT, exist_ok=True)

# ---- palette ----
SPO = "#d98a29"       # warm amber
GRPO = "#2a7fd9"      # blue
INK = "#1c2230"
MUTE = "#6b7280"
PAPER = "#ffffff"
GREEN = "#3aa64a"
RED = "#d4564a"
LIGHT = "#f1f3f7"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc, ec=None, fs=11, tc="white", rad=0.06, lw=1.5, weight="bold"):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, weight=weight)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color=INK, lw=2, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                        zorder=1, linestyle=ls)
    ax.add_patch(a)


# =====================================================================
# Diagram 1: GRPO vs SPO data flow (the central contrast)
# =====================================================================
def diagram_grpo_vs_spo():
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2))

    # ---------- GRPO ----------
    ax = axes[0]
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.2, 5.5, "GRPO  (group-based)", fontsize=15, weight="bold", color=GRPO)
    ax.text(0.2, 4.95, "generates a whole group of responses per prompt, then compares them",
            fontsize=9, color=MUTE, style="italic")

    p_prompt = box(ax, 0.3, 2.5, 1.5, 0.9, "prompt\nx", "#34405a", fs=11)
    pol = box(ax, 2.3, 2.5, 1.3, 0.9, "policy\nπ", "#7c5cbf", fs=11)
    arrow(ax, (1.8, 2.95), (2.3, 2.95))

    # group of G responses
    ys = [4.4, 3.5, 2.6, 1.2]
    labels = ["y₁ → r₁", "y₂ → r₂", "y₃ → r₃", "y_G → r_G"]
    rcolors = [GREEN, RED, GREEN, RED]
    resp_centers = []
    for yy, lab, rc in zip(ys, labels, rcolors):
        if lab.startswith("y_G"):
            ax.text(4.6, yy + 0.1, "⋮", fontsize=16, ha="center", color=MUTE)
            yy -= 0.0
        c = box(ax, 4.6, yy, 1.9, 0.62, lab, LIGHT, ec=rc, tc=INK, fs=10, lw=2)
        resp_centers.append((4.6, yy + 0.31))
        arrow(ax, (3.6, 2.95), (4.6, yy + 0.31), color=MUTE, lw=1.2, rad=0.12)

    grp = box(ax, 7.2, 2.4, 1.7, 1.3, "Group\nNorm\n(mean, std\nof G)", "#34405a", fs=9.5)
    for rc in resp_centers:
        arrow(ax, (6.5, rc[1]), (7.2, 2.95), color=MUTE, lw=1.2, rad=-0.12)

    adv = box(ax, 9.5, 2.5, 1.8, 0.9, "advantages\nA₁ … A_G", GRPO, fs=10)
    arrow(ax, (8.9, 2.95), (9.5, 2.95))

    # degeneracy callout
    ax.text(12.0, 4.2, "If all rᵢ equal\n→ all Aᵢ = 0\n→ NO learning\n   signal!",
            fontsize=9.5, color=RED, weight="bold", ha="left", va="center")
    ax.add_patch(FancyBboxPatch((11.85, 3.3), 2.0, 1.7, boxstyle="round,pad=0.05,rounding_size=0.1",
                                fc="#fdeceb", ec=RED, lw=1.5, zorder=0))

    # ---------- SPO ----------
    ax = axes[1]
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.2, 5.5, "SPO  (single-stream)", fontsize=15, weight="bold", color=SPO)
    ax.text(0.2, 4.95, "one response per prompt; a memory of past results supplies the comparison",
            fontsize=9.5, color=MUTE, style="italic")

    box(ax, 0.3, 2.6, 1.5, 0.9, "prompt\nx", "#34405a", fs=11)
    box(ax, 2.3, 2.6, 1.3, 0.9, "policy\nπ", "#7c5cbf", fs=11)
    arrow(ax, (1.8, 3.05), (2.3, 3.05))

    yb = box(ax, 4.1, 2.6, 1.5, 0.9, "response\ny → r", LIGHT, ec=GREEN, tc=INK, fs=10, lw=2)
    arrow(ax, (3.6, 3.05), (4.1, 3.05))

    # value tracker feeding baseline
    tracker = box(ax, 4.0, 0.7, 1.8, 0.95, "value tracker\nv̂(x)", SPO, fs=10)
    ax.text(4.9, 0.35, "remembers this prompt's\npast success rate", fontsize=8.2,
            color=MUTE, ha="center", va="top", style="italic")

    advs = box(ax, 7.0, 2.6, 2.2, 0.9, "advantage\nA = r − v̂(x)", SPO, fs=10.5)
    arrow(ax, (5.6, 3.05), (7.0, 3.05))
    arrow(ax, (5.8, 1.18), (7.6, 2.6), color=SPO, lw=1.6, rad=-0.25, ls="--")

    bn = box(ax, 9.7, 2.6, 1.9, 0.9, "batch norm\n(over many\nprompts)", "#34405a", fs=9.5)
    arrow(ax, (9.2, 3.05), (9.7, 3.05))

    upd = box(ax, 12.0, 2.6, 1.7, 0.9, "policy\nupdate", "#7c5cbf", fs=10.5)
    arrow(ax, (11.6, 3.05), (12.0, 3.05))

    # feedback loop tracker update
    arrow(ax, (4.85, 2.6), (4.85, 1.65), color=SPO, lw=1.6, rad=0.0, ls="--")
    ax.text(3.05, 1.15, "update\n↑ r", fontsize=8.5, color=SPO, ha="center", weight="bold")

    fig.suptitle("How GRPO and SPO differ", fontsize=17, weight="bold", y=0.99, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/d1_grpo_vs_spo.png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("saved d1")


# =====================================================================
# Diagram 2: The value tracker as a "memory dial" (Beta/EMA update)
# =====================================================================
def diagram_tracker():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # left: the running-estimate update
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(5, 7.5, "The tracker keeps ONE number per prompt", fontsize=13,
            weight="bold", ha="center", color=INK)
    ax.text(5, 6.9, "no list of past rewards — just a running estimate", fontsize=9.5,
            ha="center", color=MUTE, style="italic")

    old = box(ax, 0.5, 4.0, 2.3, 1.2, "old estimate\nv̂  = 0.40", "#34405a", fs=11)
    obs = box(ax, 0.5, 1.6, 2.3, 1.2, "new result\nr = 1 (success)", LIGHT, ec=GREEN, tc=INK, fs=10.5, lw=2)
    new = box(ax, 6.9, 2.8, 2.6, 1.4, "new estimate\nv̂ = 0.46", SPO, fs=11)

    blend = box(ax, 3.7, 2.9, 2.3, 1.2, "blend\nby a little", "#7c5cbf", fs=10.5)
    arrow(ax, (2.8, 4.6), (3.7, 3.7), color=MUTE, rad=-0.15)
    arrow(ax, (2.8, 2.2), (3.7, 3.3), color=MUTE, rad=0.15)
    arrow(ax, (6.0, 3.5), (6.9, 3.5), color=SPO)

    ax.text(5, 0.7, "v̂ ← v̂ + η · ( r − v̂ )", fontsize=13, ha="center",
            color=INK, weight="bold", family="DejaVu Sans")
    ax.text(5, 0.15, "nudge the estimate toward the newest result", fontsize=8.8,
            ha="center", color=MUTE, style="italic")

    # right: KL-adaptive forgetting
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(5, 7.5, "How fast to forget? Watch the policy.", fontsize=13,
            weight="bold", ha="center", color=INK)
    ax.text(5, 6.9, "if the policy changed a lot, old results are stale", fontsize=9.5,
            ha="center", color=MUTE, style="italic")

    # two scenarios
    box(ax, 0.5, 4.3, 4.2, 1.5,
        "policy barely changed\n→ trust old results\n→ forget slowly", LIGHT, ec=GREEN, tc=INK, fs=10, lw=2)
    box(ax, 0.5, 1.4, 4.2, 1.5,
        "policy changed a lot\n→ old results unreliable\n→ forget fast", LIGHT, ec=RED, tc=INK, fs=10, lw=2)

    # dial
    theta = np.linspace(np.pi, 0, 100)
    cx, cy, rr = 7.6, 3.0, 1.5
    ax.plot(cx + rr*np.cos(theta), cy + rr*np.sin(theta), color=INK, lw=3)
    for frac, lab, col in [(0.85, "slow", GREEN), (0.5, "", MUTE), (0.15, "fast", RED)]:
        ang = np.pi * frac
        ax.plot([cx, cx + rr*0.9*np.cos(ang)], [cy, cy + rr*0.9*np.sin(ang)],
                color=col, lw=2.5)
    ax.text(cx - rr - 0.1, cy - 0.3, "keep", fontsize=9, color=GREEN, ha="right", weight="bold")
    ax.text(cx + rr + 0.1, cy - 0.3, "forget", fontsize=9, color=RED, ha="left", weight="bold")
    ax.text(cx, cy - 0.9, "ρ  (forgetting dial)", fontsize=10, ha="center", color=INK, weight="bold")
    ax.text(cx, cy + 1.95, "driven by policy change (KL)", fontsize=8.6, ha="center", color=MUTE, style="italic")

    fig.suptitle("The KL-adaptive value tracker", fontsize=16, weight="bold", y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/d2_tracker.png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("saved d2")


# =====================================================================
# Diagram 3: Advantage = surprise (r - v_hat)
# =====================================================================
def diagram_advantage():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.2); ax.axis("off")

    ax.text(7, 4.8, "Advantage = how surprising the result was", fontsize=15,
            weight="bold", ha="center", color=INK)

    # scenario A: easy prompt solved -> small advantage
    box(ax, 0.4, 2.6, 3.0, 1.4,
        "Easy prompt\nexpected v̂ = 0.9\ngot r = 1 ✓", LIGHT, ec=MUTE, tc=INK, fs=10, lw=1.5)
    ax.text(1.9, 2.15, "A = 1 − 0.9 = +0.1", fontsize=10.5, ha="center", color=MUTE, weight="bold")
    ax.text(1.9, 1.6, "barely surprising\nsmall nudge", fontsize=9, ha="center", color=MUTE, style="italic")

    # scenario B: hard prompt solved -> big advantage
    box(ax, 5.5, 2.6, 3.0, 1.4,
        "Hard prompt\nexpected v̂ = 0.2\ngot r = 1 ✓", LIGHT, ec=GREEN, tc=INK, fs=10, lw=2)
    ax.text(7.0, 2.15, "A = 1 − 0.2 = +0.8", fontsize=10.5, ha="center", color=GREEN, weight="bold")
    ax.text(7.0, 1.6, "big positive surprise\nlearn a lot!", fontsize=9, ha="center", color=GREEN, style="italic")

    # scenario C: easy prompt failed -> big negative
    box(ax, 10.6, 2.6, 3.0, 1.4,
        "Easy prompt\nexpected v̂ = 0.9\ngot r = 0 ✗", LIGHT, ec=RED, tc=INK, fs=10, lw=2)
    ax.text(12.1, 2.15, "A = 0 − 0.9 = −0.9", fontsize=10.5, ha="center", color=RED, weight="bold")
    ax.text(12.1, 1.6, "big negative surprise\npush away!", fontsize=9, ha="center", color=RED, style="italic")

    ax.text(7, 0.6, "The tracker's estimate v̂ is the 'expectation'. Beating it is rewarded; falling short is penalized.",
            fontsize=10, ha="center", color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/d3_advantage.png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("saved d3")


# =====================================================================
# Diagram 4: Prioritized sampling curriculum (the inverted-U)
# =====================================================================
def diagram_curriculum():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    v = np.linspace(0, 1, 300)
    w = np.sqrt(v * (1 - v)) + 0.05
    ax.plot(v, w, color=SPO, lw=3, zorder=3)
    ax.fill_between(v, 0, w, color=SPO, alpha=0.12)

    ax.axvline(0.5, color=MUTE, ls=":", lw=1.5)
    ax.annotate("MEDIUM prompts\nget the most practice\n(most to learn)",
                xy=(0.5, np.sqrt(0.25) + 0.05), xytext=(0.5, 0.30),
                ha="center", fontsize=10, color=INK, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.annotate("already mastered\n(skip mostly)", xy=(0.95, np.sqrt(0.95*0.05)+0.05),
                xytext=(0.86, 0.42), ha="center", fontsize=9, color=MUTE,
                arrowprops=dict(arrowstyle="-|>", color=MUTE))
    ax.annotate("too hard right now\n(skip mostly)", xy=(0.05, np.sqrt(0.05*0.95)+0.05),
                xytext=(0.14, 0.42), ha="center", fontsize=9, color=MUTE,
                arrowprops=dict(arrowstyle="-|>", color=MUTE))

    ax.text(0.012, 0.07, "tiny floor (ε) keeps\nevery prompt reachable", fontsize=8.2,
            color=GREEN, style="italic", va="bottom")

    ax.set_xlabel("prompt's current success rate  v̂", fontsize=11)
    ax.set_ylabel("sampling weight  w", fontsize=11)
    ax.set_title("Adaptive curriculum: practice what's learnable right now",
                 fontsize=14, weight="bold", color=INK, pad=12)
    ax.set_ylim(0, 0.62); ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/d4_curriculum.png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("saved d4")


# =====================================================================
# Diagram 5: The training loop (Algorithm 1) as numbered steps
# =====================================================================
def diagram_loop():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5, 9.6, "One SPO training step", fontsize=16, weight="bold", ha="center", color=INK)

    steps = [
        ("1", "Pick prompts", "by curriculum weight", SPO),
        ("2", "Run one response each", "single stream → reward r", "#7c5cbf"),
        ("3", "Compute advantage", "A = r − v̂(x)", GREEN),
        ("4", "Update the tracker", "fold r into v̂(x)", SPO),
        ("5", "Normalize across batch", "stabilize the scale", "#34405a"),
        ("6", "Update the policy", "PPO-Clip step", "#7c5cbf"),
    ]
    # circular-ish vertical layout in two columns
    positions = [(1.2, 7.6), (1.2, 5.2), (1.2, 2.8),
                 (6.0, 2.8), (6.0, 5.2), (6.0, 7.6)]
    centers = []
    for (num, title, sub, col), (x, y) in zip(steps, positions):
        c = Circle((x, y + 0.0), 0.45, fc=col, ec=col, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, num, ha="center", va="center", color="white", fontsize=14, weight="bold", zorder=4)
        ax.text(x + 0.75, y + 0.18, title, fontsize=11.5, weight="bold", color=INK, va="center")
        ax.text(x + 0.75, y - 0.28, sub, fontsize=9, color=MUTE, va="center", style="italic")
        centers.append((x, y))

    # arrows 1->2->3 (down left), 3->4 (across), 4->5->6 (up right)
    arrow(ax, (centers[0][0], centers[0][1]-0.5), (centers[1][0], centers[1][1]+0.5), color=MUTE)
    arrow(ax, (centers[1][0], centers[1][1]-0.5), (centers[2][0], centers[2][1]+0.5), color=MUTE)
    arrow(ax, (centers[2][0]+0.5, centers[2][1]), (centers[3][0]-0.5, centers[3][1]), color=MUTE)
    arrow(ax, (centers[3][0], centers[3][1]+0.5), (centers[4][0], centers[4][1]-0.5), color=MUTE)
    arrow(ax, (centers[4][0], centers[4][1]+0.5), (centers[5][0], centers[5][1]-0.5), color=MUTE)
    # loop back 6 -> 1
    arrow(ax, (centers[5][0], centers[5][1]+0.5), (centers[0][0], centers[0][1]+0.5),
          color=SPO, lw=2, rad=-0.4, ls="--")
    ax.text(3.6, 8.7, "repeat", fontsize=10, color=SPO, style="italic", weight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT}/d5_loop.png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("saved d5")


# =====================================================================
# Diagram 6: Synchronization bottleneck (why group-free is faster)
# =====================================================================
def diagram_bottleneck():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    rng = np.random.default_rng(2)

    times_group = [90, 444, 508, 74, 114, 117, 87, 409]
    ax = axes[0]
    colors = [RED if t > 250 else GRPO for t in times_group]
    ax.barh(range(8), times_group, color=colors)
    ax.axvline(max(times_group), color=RED, ls="--", lw=2)
    ax.text(max(times_group) - 10, 6.4, f"everyone waits\nfor {max(times_group)}s",
            color=RED, ha="right", fontsize=9.5, weight="bold")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"resp {i+1}" for i in range(8)], fontsize=8)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_title("GRPO: the group waits for its slowest member",
                 fontsize=11.5, weight="bold", color=GRPO)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    pool = sorted([90, 112, 94, 114, 74, 117, 87, 95, 444, 508, 409, 130], )
    take = 8
    colors = [SPO if i < take else "#d8dde6" for i in range(len(pool))]
    ax.barh(range(len(pool)), pool, color=colors)
    ax.axvline(pool[take-1], color=SPO, ls="--", lw=2)
    ax.text(pool[take-1] + 12, 1.0, f"collect first {take}\nby {pool[take-1]}s",
            color=SPO, ha="left", fontsize=9.5, weight="bold")
    ax.set_yticks([]); ax.set_xlabel("time (s)", fontsize=10)
    ax.set_title("SPO: take the first to finish, drop stragglers",
                 fontsize=11.5, weight="bold", color=SPO)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Why group-free is faster in agentic settings", fontsize=15, weight="bold", y=1.04, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/d6_bottleneck.png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("saved d6")


if __name__ == "__main__":
    diagram_grpo_vs_spo()
    diagram_tracker()
    diagram_advantage()
    diagram_curriculum()
    diagram_loop()
    diagram_bottleneck()
    print("\nAll diagrams saved to", OUT)
