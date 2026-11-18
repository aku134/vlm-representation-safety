"""Stage 1 of eval: generations (+ benign perplexity) for base vs circuit-broken.

Loads ONE model. With an adapter, base = `disable_adapter()` and cb = adapter on,
so a T4 never holds two 2B copies. The judge (Llama Guard) runs afterwards in
a separate process: `src/judge.py`, which also writes eval_summary.json.

Conditions (fixed — never change between runs being compared):
    benign_clean       benign image + carrier
    harmful_clean      harmful image + carrier
    harmful_blur4x     harmful image, 4x downscale/upscale blur
    harmful_occlude20  harmful image, 20% gray patches
    harmful_text       harmful prompt as plain text, no image

Benign quality: perplexity of each benign response under the frozen base
model (adapter off), conditioned on the same image + carrier prompt.

Usage:
    # baseline (no adapter)
    python src/eval_cb.py --run-dir out/runs/base --base-only
    # a trained run: reuses base generations from out/runs/base if they match
    python src/eval_cb.py --run-dir out/runs/R0 --base-from out/runs/base
    python src/judge.py --run-dir out/runs/R0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from perturbations import downscale_blur, occlude
from vlm_common import (
    DEVICE,
    ROOT,
    classify_generation,
    load_base,
    load_circuit_broken,
    load_processor,
)

CONDITIONS = [
    # name, label, kind, image transform
    ("benign_clean", "benign", "image", lambda im: im),
    ("harmful_clean", "harmful", "image", lambda im: im),
    ("harmful_blur4x", "harmful", "image", lambda im: downscale_blur(im, 4)),
    ("harmful_occlude20", "harmful", "image", lambda im: occlude(im, 0.20)),
    ("harmful_text", "harmful", "text", None),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def user_message(item, kind, image):
    if kind == "text":
        return {"role": "user", "content": [{"type": "text", "text": item["instruction"]}]}
    return {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": item["carrier"]}],
    }


def encode(processor, messages_list, images):
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    kw = dict(text=texts, padding=True, return_tensors="pt")
    if images:
        kw["images"] = images
    enc = processor(**kw)
    return {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in enc.items()}


@torch.no_grad()
def generate_batch(model, processor, jobs, use_adapter, max_new_tokens):
    """jobs: list of (item, kind, image-or-None). Greedy decoding, left padding."""
    msgs = [[user_message(it, kind, img)] for it, kind, img in jobs]
    images = [img for _, kind, img in jobs if kind == "image"]
    enc = encode(processor, msgs, images)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    if use_adapter or not hasattr(model, "disable_adapter"):
        out = model.generate(**enc, **gen_kwargs)
    else:
        with model.disable_adapter():
            out = model.generate(**enc, **gen_kwargs)
    new = out[:, enc["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)


@torch.no_grad()
def response_perplexity(model, processor, item, image, response):
    """PPL of `response` given image+carrier, under the base model (adapter off)."""
    if not response.strip():
        return None
    user = user_message(item, "image", image)
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": response}]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    p_enc = processor(text=[prompt], images=[image], return_tensors="pt")
    f_enc = processor(text=[full], images=[image], return_tensors="pt")
    f_enc = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in f_enc.items()}
    start = p_enc["input_ids"].shape[1]
    ids = f_enc["input_ids"][0]
    end = ids.shape[0]
    # Drop the trailing end-of-turn tokens the template appends after the text.
    while end > start and processor.tokenizer.decode(ids[end - 1]).strip() in ("", "<|im_end|>"):
        end -= 1
    if end <= start:
        return None

    def _logits():
        return model(**f_enc, use_cache=False).logits[0]

    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            logits = _logits()
    else:
        logits = _logits()
    # token t is predicted by logits at t-1
    lp = torch.log_softmax(logits[start - 1 : end - 1].float(), dim=-1)
    tgt = ids[start:end]
    nll = -lp.gather(-1, tgt[:, None]).squeeze(-1).mean()
    return float(math.exp(min(nll.item(), 50.0)))


@torch.no_grad()
def adapter_logit_delta(model, processor, item):
    """Sanity check: mean |logit(adapter on) - logit(adapter off)| on one prompt."""
    if not hasattr(model, "disable_adapter"):
        return None
    img = Image.open(ROOT / item["image"]).convert("RGB")
    enc = encode(processor, [[user_message(item, "image", img)]], [img])
    on = model(**enc, use_cache=False).logits[0, -1].float()
    with model.disable_adapter():
        off = model(**enc, use_cache=False).logits[0, -1].float()
    return float((on - off).abs().mean())


def eval_items(manifest, limit):
    harmful = [x for x in manifest if x["label"] == "harmful"][:limit]
    benign = [x for x in manifest if x["label"] == "benign"][:limit]
    return harmful, benign


def config_fingerprint(cfg):
    keys = ("manifest", "limit", "max_new_tokens", "conditions", "item_ids", "gen_batch_size")
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main(args):
    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "eval"
    gen_path = eval_dir / "generations.json"
    if gen_path.exists() and not args.overwrite_eval:
        raise SystemExit(f"{gen_path} exists; pass --overwrite-eval to regenerate")
    eval_dir.mkdir(parents=True, exist_ok=True)

    limit = 4 if args.smoke else args.limit
    manifest = json.load(open(ROOT / args.manifest))
    harmful, benign = eval_items(manifest, limit)
    print(f"eval items: {len(harmful)} harmful, {len(benign)} benign (limit {limit})")
    if len(benign) < limit or len(harmful) < limit:
        print(f"[note] eval split has fewer than {limit} of a class; using all available")

    cfg = dict(
        manifest=args.manifest,
        limit=limit,
        max_new_tokens=args.max_new_tokens,
        conditions=[c[0] for c in CONDITIONS],
        item_ids=sorted(x["id"] for x in harmful + benign),
        seed=args.seed,
        gen_batch_size=args.gen_batch_size,
    )
    cfg["fingerprint"] = config_fingerprint(cfg)

    processor = load_processor()
    processor.tokenizer.padding_side = "left"

    adapter = None if args.base_only else Path(args.adapter or run_dir / "adapter")
    if adapter is not None:
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"no adapter at {adapter}; use --base-only for the baseline")
        model = load_circuit_broken(adapter)
        modes = [("cb", True)]
    else:
        model = load_base()
        modes = [("base", False)]

    # Reuse base generations from the baseline run when the eval config matches
    # (greedy decoding, same items) — halves eval time for every CB run.
    base_records = None
    if adapter is not None and args.base_from:
        bpath = Path(args.base_from) / "eval" / "generations.json"
        bcfg_path = Path(args.base_from) / "eval" / "eval_config.json"
        if bpath.exists() and bcfg_path.exists():
            bcfg = json.load(open(bcfg_path))
            if bcfg.get("fingerprint") == cfg["fingerprint"]:
                base_records = [r for r in json.load(open(bpath)) if r["model"] == "base"]
                print(f"reusing {len(base_records)} base generations from {bpath}")
            else:
                print("[warn] base run eval config differs; regenerating base here")
        else:
            print(f"[warn] {bpath} missing; regenerating base here")
    if adapter is not None and base_records is None:
        modes = [("base", False)] + modes

    if adapter is not None:
        delta = adapter_logit_delta(model, processor, harmful[0])
        cfg["adapter_logit_delta"] = delta
        print(f"sanity: mean |logit(on)-logit(off)| = {delta:.4g}")
        if delta is not None and delta < 1e-4:
            print("[warn] adapter on/off give identical logits — adapter is not active!")
    cfg["adapter"] = None if adapter is None else str(adapter)
    cfg["modes"] = [m for m, _ in modes] + (["base(reused)"] if base_records else [])

    records = list(base_records or [])
    for cname, label, kind, fn in CONDITIONS:
        items = harmful if label == "harmful" else benign
        jobs = []
        for it in items:
            img = None
            if kind == "image":
                img = fn(Image.open(ROOT / it["image"]).convert("RGB"))
            jobs.append((it, kind, img))
        for mname, use_adapter in modes:
            for i in range(0, len(jobs), args.gen_batch_size):
                chunk = jobs[i : i + args.gen_batch_size]
                texts = generate_batch(model, processor, chunk, use_adapter, args.max_new_tokens)
                for (it, _, _), text in zip(chunk, texts):
                    records.append(
                        dict(
                            id=it["id"],
                            label=label,
                            condition=cname,
                            model=mname,
                            request=it["instruction"],
                            generation=text,
                            regex_verdict=classify_generation(text),
                        )
                    )
                print(f"{cname:>18} {mname:>4} {i + len(chunk):>3}/{len(jobs)}", flush=True)
                torch.cuda.empty_cache()

    # Benign perplexity under the frozen base model.
    by_id = {x["id"]: x for x in benign}
    for r in records:
        if r["condition"] == "benign_clean" and r.get("ppl_base") is None:
            it = by_id[r["id"]]
            img = Image.open(ROOT / it["image"]).convert("RGB")
            r["ppl_base"] = response_perplexity(model, processor, it, img, r["generation"])
    print("benign perplexity done")

    with open(eval_dir / "eval_config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    with open(gen_path, "w") as f:
        json.dump(records, f, indent=1)
    print(f"wrote {gen_path} ({len(records)} records). Next: python src/judge.py --run-dir {run_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--adapter", default=None, help="default: <run-dir>/adapter")
    ap.add_argument("--base-only", action="store_true", help="evaluate the base model only")
    ap.add_argument("--base-from", default="out/runs/base", help="reuse base generations from this run")
    ap.add_argument("--manifest", default="data/manifest_eval.json")
    ap.add_argument("--limit", type=int, default=50, help="examples per class")
    ap.add_argument("--smoke", action="store_true", help="4 examples per condition")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite-eval", action="store_true")
    args = ap.parse_args()
    sys.exit(main(args))
