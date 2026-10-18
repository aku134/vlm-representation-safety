"""Extract and cache base-model hidden states for probe training.

For every example in a manifest, runs one forward pass through base
Qwen2-VL-2B and saves hidden states at the candidate layers for the last
`n_tokens` prompt positions (the positions where the model has fused image
and text and 'decided' what the request means).

Output: outputs/activations_<split>.npz
    keys: "<id>|layer<k>" -> (n_tokens, hidden) float16 arrays
    plus "labels" json: {id: label}
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image

from vlm_common import ROOT, build_inputs, load_base, load_processor

# Qwen2-VL-2B has 28 layers; sweep around the depth fractions the reference
# used for Llama (10/32, 15/32, 20/32) plus one later layer.
CANDIDATE_LAYERS = [9, 13, 18, 22]


@torch.no_grad()
def main(args):
    processor = load_processor()
    model = load_base()

    manifest = json.load(open(ROOT / "data" / f"manifest_{args.split}.json"))
    if args.limit:
        harmful = [x for x in manifest if x["label"] == "harmful"][: args.limit]
        benign = [x for x in manifest if x["label"] == "benign"][: args.limit]
        manifest = harmful + benign

    store: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for i, item in enumerate(manifest):
        img = Image.open(ROOT / item["image"]).convert("RGB")
        inputs = build_inputs(processor, img, item["carrier"])
        out = model(**inputs, output_hidden_states=True)
        for layer in CANDIDATE_LAYERS:
            hs = out.hidden_states[layer][0]          # (seq, hidden)
            tail = hs[-args.n_tokens :].float().cpu().numpy().astype(np.float16)
            store[f"{item['id']}|layer{layer}"] = tail
        labels[item["id"]] = item["label"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"{i + 1}/{len(manifest)}", flush=True)

    out_path = ROOT / "outputs" / f"activations_{args.split}.npz"
    out_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(out_path, labels=json.dumps(labels), **store)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(labels)} examples)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--n-tokens", type=int, default=8, help="last-k prompt positions")
    ap.add_argument("--limit", type=int, default=0, help="per-class cap (0 = all)")
    args = ap.parse_args()
    main(args)
