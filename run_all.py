"""Run the entire reproduction: tests + all four experiments.

    python run_all.py
"""

import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, label):
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    r = subprocess.run([sys.executable] + cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"!! {label} failed with code {r.returncode}")
        sys.exit(r.returncode)


if __name__ == "__main__":
    run(["tests/test_spo.py"], "Unit tests (verify core math)")
    run(["experiments/run_comparison.py"], "Exp 1: SPO vs GRPO (Table 1 / Fig 3a)")
    run(["experiments/run_ablations.py"], "Exp 2: Ablations (Fig 2)")
    run(["experiments/run_variance.py"], "Exp 3: Variance decomposition (Eq 12)")
    run(["experiments/run_throughput.py"], "Exp 4: Agentic throughput (Fig 4 & 5)")
    print("\nAll done. Figures are in ./figures/")
