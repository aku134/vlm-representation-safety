"""Print one markdown table comparing eval_summary.json across run folders.

    python scripts/compare_runs.py out/runs/base out/runs/R0 out/runs/R1 ...

For the `base` folder the base-model rows are shown; for every other run the
circuit-broken (`cb`) rows. A run folder containing FAILED.txt is listed as
failed with the reason from that file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HARM_CONDS = [
    ("harmful_clean", "clean"),
    ("harmful_blur4x", "blur 4x"),
    ("harmful_occlude20", "occl 20%"),
    ("harmful_text", "text-only"),
]


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def run_settings(run_dir: Path) -> str:
    p = run_dir / "args.json"
    if not p.exists():
        return "base model"
    a = json.load(open(p))
    parts = [
        f"{a.get('cb_loss', '?')}",
        f"T={a.get('target_layers', '?')}",
        f"R={a.get('retain_layers', 'targets')}",
    ]
    if a.get("kl_coeff"):
        parts.append(f"KL={a['kl_coeff']}")
    if a.get("normalize_losses"):
        parts.append("norm")
    if "pairs" in str(a.get("manifest", "")):
        parts.append("pairs")
    if a.get("smoke"):
        parts.append("SMOKE")
    return ", ".join(parts)


def table(run_dirs):
    head = (
        ["run", "settings"]
        + [f"judge harm {lab}" for _, lab in HARM_CONDS]
        + ["benign comply", "benign incoh", "benign ppl (mean)", "benign ppl>thr", "n harm/ben"]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for rd in map(Path, run_dirs):
        name = rd.name
        failed = rd / "FAILED.txt"
        if failed.exists():
            reason = failed.read_text().strip().splitlines()[0][:80]
            lines.append(f"| {name} | FAILED: {reason} |" + " |" * (len(head) - 2))
            continue
        sp = rd / "eval_summary.json"
        if not sp.exists():
            lines.append(f"| {name} | (no eval_summary.json) |" + " |" * (len(head) - 2))
            continue
        summ = json.load(open(sp))
        model = "base" if name == "base" else "cb"
        rows = {r["condition"]: r for r in summ["rows"] if r["model"] == model}
        ben = rows.get("benign_clean", {})
        cells = [name, run_settings(rd)]
        cells += [pct(rows.get(c, {}).get("judge_harmful_rate")) for c, _ in HARM_CONDS]
        cells += [
            pct(ben.get("regex_comply_rate")),
            pct(ben.get("regex_incoherent_rate")),
            "–" if ben.get("ppl_mean") is None else f"{ben['ppl_mean']:.2f}",
            pct(ben.get("ppl_frac_above")),
            f"{rows.get('harmful_clean', {}).get('n', '–')}/{ben.get('n', '–')}",
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    thr = None
    for rd in map(Path, run_dirs):
        sp = rd / "eval_summary.json"
        if sp.exists():
            thr = json.load(open(sp)).get("ppl_threshold")
            break
    note = (
        f"\njudge = Llama-Guard-3-1B unsafe rate on (rendered request, response); lower is safer. "
        f"benign comply/incoh = regex classifier. ppl = perplexity of the benign response under the "
        f"frozen base model; ppl>thr = fraction above {thr} (empty responses count as above)."
    )
    return "\n".join(lines) + "\n" + note


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=None, help="also write the table to this file")
    args = ap.parse_args()
    t = table(args.run_dirs)
    print(t)
    if args.out:
        Path(args.out).write_text(t + "\n")
