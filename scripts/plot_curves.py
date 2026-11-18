"""Plot retain loss, CB loss and CB cosine over steps for several runs.

    python scripts/plot_curves.py out/runs/R0 out/runs/R3 --out results/cb_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("retain_loss_raw", "retain loss (raw L2)"),
    ("cb_loss_raw", "CB loss (raw)"),
    ("cb_cosine", "cosine(LoRA, frozen) on CB batch"),
]


def main(args):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.4))
    for rd in map(Path, args.run_dirs):
        log = json.load(open(rd / "train_log.json"))["log"]
        steps = [e["step"] for e in log]
        for ax, (key, title) in zip(axes, PANELS):
            ys = [e.get(key) for e in log]
            pts = [(s, y) for s, y in zip(steps, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), label=rd.name, linewidth=1.6)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("step")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[-1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="results/cb_curves.png")
    main(ap.parse_args())
