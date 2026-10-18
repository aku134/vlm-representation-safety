"""Before/after evaluation: base vs circuit-broken Qwen2-VL on held-out images.

Loads ONE PeftModel and toggles the adapter (disable_adapter = base,
adapter on = circuit-broken) so a T4 16GB does not OOM from two 2B copies.

Outputs
-------
outputs/eval_generations.json
outputs/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from perturbations import iter_perturbations
from vlm_common import (
    ROOT,
    classify_generation,
    generate,
    load_circuit_broken,
    load_processor,
    load_base,
)


@torch.no_grad()
def generate_mode(model, processor, image, carrier, use_adapter: bool, max_new_tokens: int):
    """If model is PeftModel, toggle adapter; else plain generate."""
    if use_adapter or not hasattr(model, "disable_adapter"):
        return generate(model, processor, image, carrier, max_new_tokens)
    with model.disable_adapter():
        return generate(model, processor, image, carrier, max_new_tokens)


def main(args):
    processor = load_processor()
    adapter = Path(args.adapter)

    if adapter.exists() and (adapter / "adapter_config.json").exists():
        print(f"loading single PeftModel from {adapter} (base via disable_adapter)")
        model = load_circuit_broken(adapter)
        modes = [("base", False), ("circuit_broken", True)]
    else:
        print(f"[warn] adapter not found at {adapter} — base only")
        model = load_base()
        modes = [("base", True)]  # no adapter; flag unused

    manifest = json.load(open(ROOT / "data" / "manifest_eval.json"))
    harmful = [x for x in manifest if x["label"] == "harmful"][: args.limit]
    benign = [x for x in manifest if x["label"] == "benign"][: args.limit]
    print(f"evaluating {len(harmful)} harmful + {len(benign)} benign examples")

    perturbs = (
        list(iter_perturbations())
        if not args.clean_only
        else [("clean", "none", lambda im: im)]
    )

    records = []
    for label, items in (("harmful", harmful), ("benign", benign)):
        conds = perturbs if label == "harmful" else [("clean", "none", lambda im: im)]
        for item in items:
            img0 = Image.open(ROOT / item["image"]).convert("RGB")
            for pname, severity, fn in conds:
                img = fn(img0)
                for mname, use_adapter in modes:
                    text = generate_mode(
                        model, processor, img, item["carrier"],
                        use_adapter=use_adapter,
                        max_new_tokens=args.max_new_tokens,
                    )
                    verdict = classify_generation(text)
                    records.append(
                        dict(
                            id=item["id"],
                            label=label,
                            perturbation=pname,
                            severity=severity,
                            model=mname,
                            verdict=verdict,
                            generation=text[:600],
                        )
                    )
                    print(
                        f"{item['id']:>12} {pname:>8}/{severity:<9} "
                        f"{mname:>14} -> {verdict}"
                    )
                torch.cuda.empty_cache()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "eval_generations.json", "w") as f:
        json.dump(records, f, indent=1)

    summary = {}
    for r in records:
        key = (r["model"], r["label"], r["perturbation"], r["severity"])
        summary.setdefault(key, []).append(r["verdict"])
    rows = []
    for (model_name, label, pname, severity), verdicts in sorted(summary.items()):
        n = len(verdicts)
        comply = sum(v == "comply" for v in verdicts) / n
        refuse = sum(v == "refusal" for v in verdicts) / n
        incoh = sum(v == "incoherent" for v in verdicts) / n
        rows.append(
            dict(
                model=model_name,
                label=label,
                perturbation=pname,
                severity=severity,
                n=n,
                comply_rate=round(comply, 3),
                refusal_rate=round(refuse, 3),
                incoherent_rate=round(incoh, 3),
            )
        )
        print(
            f"{model_name:>14} {label:>8} {pname:>8}/{severity:<9} "
            f"comply={comply:.2f} refuse={refuse:.2f} incoherent={incoh:.2f}"
        )
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out_dir / 'eval_generations.json'} and eval_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        default=str(ROOT / "out" / "qwen2vl-cb-v2" / "adapter"),
    )
    ap.add_argument("--limit", type=int, default=20, help="examples per class")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()
    main(args)
