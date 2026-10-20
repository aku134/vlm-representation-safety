"""Perturbation robustness: probe vs OCR baseline on held-out harmful images.

For each perturbation x severity:
  * probe detection rate  — run the perturbed image through base Qwen2-VL,
    score activations at the probe layer, compare to the tuned threshold;
  * OCR detection rate    — pytesseract reads the perturbed image; detection =
    extracted text sufficiently matches the harmful instruction (the
    input-side defense an attacker is trying to beat).

Outputs
-------
outputs/perturb_results.json
outputs/perturb_chart.png      grouped bars, probe vs OCR per condition
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from PIL import Image

from ocr_common import ocr_detects_instruction as ocr_detects
from perturbations import iter_perturbations
from vlm_common import ROOT, hidden_states_at, load_base, load_processor


def main(args):
    probe = joblib.load(ROOT / "outputs" / "probe.joblib")
    clf, layer, threshold = probe["clf"], probe["layer"], probe["threshold"]
    print(f"probe: layer {layer}, threshold {threshold:.4f}")

    processor = load_processor()
    model = load_base()

    manifest = json.load(open(ROOT / "data" / "manifest_eval.json"))
    harmful = [x for x in manifest if x["label"] == "harmful"][: args.limit]
    print(f"{len(harmful)} held-out harmful examples")

    # OCR results: prefer the cache from ocr_baseline.py (runs without torch,
    # and outside this process if the Vision service is unavailable here).
    ocr_cache: dict[tuple, bool] = {}
    ocr_cache_path = ROOT / "outputs" / "ocr_results.json"
    if ocr_cache_path.exists():
        for r in json.load(open(ocr_cache_path)):
            ocr_cache[(r["id"], r["perturbation"], r["severity"])] = r["detected"]
        print(f"using cached OCR results ({len(ocr_cache)} entries)")

    results = []
    for pname, severity, fn in iter_perturbations():
        probe_hits, ocr_hits = 0, 0
        for item in harmful:
            img = fn(Image.open(ROOT / item["image"]).convert("RGB"))
            hs = hidden_states_at(model, processor, img, item["carrier"], layer)
            tail = hs[-args.n_tokens :].numpy().astype(np.float32)
            score = float(clf.predict_proba(tail)[:, 1].max())
            probe_hits += score >= threshold
            key = (item["id"], pname, severity)
            if key in ocr_cache:
                ocr_hits += ocr_cache[key]
            else:
                ocr_hits += ocr_detects(img, item["instruction"])
        n = len(harmful)
        rec = dict(
            perturbation=pname, severity=severity, n=n,
            probe_rate=round(probe_hits / n, 3), ocr_rate=round(ocr_hits / n, 3),
        )
        results.append(rec)
        print(f"{pname:>8}/{severity:<9} probe {rec['probe_rate']:.2f}  ocr {rec['ocr_rate']:.2f}")

    out_dir = ROOT / "outputs"
    with open(out_dir / "perturb_results.json", "w") as f:
        json.dump(results, f, indent=1)

    # ---- chart ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['perturbation']}\n{r['severity']}" for r in results]
    x = np.arange(len(results))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.9), 4.5))
    ax.bar(x - w / 2, [r["probe_rate"] for r in results], w,
           label="Activation probe (representation-level)", color="#2d7ff9")
    ax.bar(x + w / 2, [r["ocr_rate"] for r in results], w,
           label="OCR keyword filter (input-level)", color="#c44")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Detection rate on held-out harmful images")
    ax.set_ylim(0, 1.05)
    ax.set_title("Harmful typographic-image detection under perturbation\n"
                 f"(base Qwen2-VL-2B activations, layer {layer}, n={results[0]['n']} per condition)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perturb_chart.png", dpi=150)
    print(f"wrote {out_dir / 'perturb_chart.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30, help="harmful examples per condition")
    ap.add_argument("--n-tokens", type=int, default=8)
    args = ap.parse_args()
    main(args)
