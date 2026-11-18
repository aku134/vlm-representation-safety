"""Build the typographic-image dataset for multimodal circuit-breaker training.

Takes Gray Swan's text data (circuit_breakers_train.json, XSTest completions,
optionally UltraChat) and renders every prompt as a typographic image: the
instruction is drawn into the image, and the text channel carries only an
innocuous carrier prompt. Harmful and benign examples use the *identical*
rendering style so nothing about the image style leaks the label.

Outputs
-------
images/<split>/<id>.png                 rendered typographic images
data/manifest_train.json                training examples (harmful + benign)
data/manifest_eval.json                 held-out examples, never trained on
data/dataset_colab.zip                  images + manifests, ready for Colab

Manifest entry schema:
    {
      "id": "harm_000123",
      "label": "harmful" | "benign",
      "source": "circuit_breakers" | "xstest" | "ultrachat",
      "instruction": <text rendered into the image>,
      "carrier": <text prompt sent alongside the image>,
      "response": <assistant response used in training>,
          # harmful -> the harmful completion (circuit-breaker target)
          # benign  -> the compliant answer      (retain target)
      "refusal": <refusal text or null>,  # llama3_output, retain alternative
      "image": "images/train/harm_000123.png"
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = ROOT / "images"

CARRIERS = [
    "Follow the instructions written in the image.",
    "Please carry out the request shown in this image.",
    "Do what the image says.",
    "Read the image and complete the task it describes.",
]

# Fonts available on macOS out of the box; fall back to PIL default.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]

IMG_SIZE = 448  # keeps Qwen2-VL vision token count modest


def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is not None:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sample_style(rng: random.Random) -> dict:
    return dict(
        font_path=rng.choice(FONT_CANDIDATES + [None]),
        font_size=rng.randint(18, 26),
        bg=rng.choice([(255, 255, 255), (245, 245, 240), (235, 240, 245)]),
    )


def render_typographic(text: str, out_path: Path | None, rng: random.Random | None, style: dict | None = None):
    """Render `text` onto a plain background, wrapped and roughly centered.

    Style comes from `rng` (same draw order as always) unless `style` is given.
    out_path=None only samples the style (used to replay the build). Returns (style, image).
    """
    style = style or sample_style(rng)
    if out_path is None:
        return style, None
    font_path, font_size, bg = style["font_path"], style["font_size"], tuple(style["bg"])
    font = load_font(font_path, font_size)
    fg = (20, 20, 20)

    # Wrap to fit: estimate chars per line from font size.
    chars_per_line = max(18, int(IMG_SIZE / (font_size * 0.55)))
    # Truncate very long prompts; the instruction core is at the front.
    text = text.strip()
    if len(text) > 600:
        text = text[:600]
    lines = textwrap.wrap(text, width=chars_per_line)
    line_height = font_size + 6
    max_lines = (IMG_SIZE - 40) // line_height
    lines = lines[:max_lines]

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)
    total_h = len(lines) * line_height
    y = max(20, (IMG_SIZE - total_h) // 2)
    for line in lines:
        draw.text((24, y), line, fill=fg, font=font)
        y += line_height

    img.save(out_path, format="PNG")
    return style, img


def build(n_harmful: int, n_benign: int, eval_frac: float, seed: int, use_ultrachat: bool, dry: bool = False):
    """dry=True replays the RNG without writing anything; returns {id: style+carrier}."""
    rng = random.Random(seed)

    # ---------- harmful: Gray Swan circuit breaker pairs ----------
    with open(DATA / "circuit_breakers_train.json") as f:
        cb = json.load(f)
    rng.shuffle(cb)
    harmful = []
    for i, d in enumerate(cb[:n_harmful]):
        harmful.append(
            dict(
                id=f"harm_{i:06d}",
                label="harmful",
                source="circuit_breakers",
                instruction=d["prompt"],
                response=d["output"],           # harmful completion -> CB loss
                refusal=d.get("llama3_output"), # refusal -> retain alternative
            )
        )

    # ---------- benign: XSTest full compliance (+ UltraChat) ----------
    with open(DATA / "xstest_v2_completions_gpt4_gpteval.csv", newline="") as f:
        xs = [r for r in csv.DictReader(f) if r["final_label"] == "1_full_compliance"]
    rng.shuffle(xs)

    benign = []
    for i, r in enumerate(xs):
        benign.append(
            dict(
                id=f"ben_{i:06d}",
                label="benign",
                source="xstest",
                instruction=r["prompt"],
                response=r["completion"],
                refusal=None,
            )
        )

    if use_ultrachat and len(benign) < n_benign:
        try:
            from datasets import load_dataset

            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
            need = n_benign - len(benign)
            for j, ex in enumerate(ds):
                msgs = ex["messages"]
                if len(msgs) < 2:
                    continue
                prompt, reply = msgs[0]["content"], msgs[1]["content"]
                if len(prompt) < 20 or len(prompt) > 600:
                    continue
                benign.append(
                    dict(
                        id=f"ben_u{j:06d}",
                        label="benign",
                        source="ultrachat",
                        instruction=prompt,
                        response=reply,
                        refusal=None,
                    )
                )
                if len(benign) >= n_benign:
                    break
        except Exception as e:  # offline, missing package, etc.
            print(f"[warn] UltraChat unavailable ({e}); continuing with XSTest only")

    benign = benign[:n_benign]
    rng.shuffle(benign)

    # ---------- split, render, write manifests ----------
    def split(items):
        n_eval = max(1, int(len(items) * eval_frac))
        return items[n_eval:], items[:n_eval]

    harm_train, harm_eval = split(harmful)
    ben_train, ben_eval = split(benign)

    manifests = {
        "train": harm_train + ben_train,
        "eval": harm_eval + ben_eval,
    }

    styles = {}
    for split_name, items in manifests.items():
        out_dir = IMAGES / split_name
        if not dry:
            out_dir.mkdir(parents=True, exist_ok=True)
        rng.shuffle(items)
        for item in items:
            img_rel = f"images/{split_name}/{item['id']}.png"
            style, _ = render_typographic(item["instruction"], None if dry else ROOT / img_rel, rng)
            item["image"] = img_rel
            item["carrier"] = rng.choice(CARRIERS)
            styles[item["id"]] = dict(style, carrier=item["carrier"], split=split_name)
        if dry:
            continue
        with open(DATA / f"manifest_{split_name}.json", "w") as f:
            json.dump(items, f, indent=1)
        n_h = sum(1 for x in items if x["label"] == "harmful")
        print(f"{split_name}: {len(items)} examples ({n_h} harmful, {len(items) - n_h} benign)")

    if dry:
        return styles

    # ---------- archive for Colab ----------
    zip_path = DATA / "dataset_colab.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for split_name, items in manifests.items():
            z.write(DATA / f"manifest_{split_name}.json", f"data/manifest_{split_name}.json")
            for item in items:
                z.write(ROOT / item["image"], item["image"])
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def add_minimal_pairs(args) -> None:
    """Render benign twins with the exact style + carrier of their harmful twin.

    Reads data/minimal_pairs.json (reviewed by a human), recovers each harmful
    image's font/size/background by replaying the seeded build, verifies the
    replay by re-rendering harmful images and comparing pixels, then writes
    images/train/twin_<harm_id>.png and data/manifest_train_pairs.json
    (= manifest_train.json + twins). The eval manifest is not touched.
    """
    import numpy as np

    pairs_path = DATA / "minimal_pairs.json"
    if not pairs_path.exists():
        raise SystemExit(
            f"{pairs_path} missing. Run scripts/make_minimal_pairs.py, review the file, then rerun."
        )
    pairs = json.load(open(pairs_path))
    train = json.load(open(DATA / "manifest_train.json"))
    by_id = {x["id"]: x for x in train}
    styles = build(args.n_harmful, args.n_benign, args.eval_frac, args.seed, not args.no_ultrachat, dry=True)

    # Verify the replay reproduces existing images exactly.
    checked = 0
    for item in [x for x in train if x["label"] == "harmful"][:10]:
        st = styles[item["id"]]
        assert st["carrier"] == item["carrier"], f"carrier mismatch for {item['id']}: replay is off"
        _, img = render_typographic(item["instruction"], DATA / "_replay_check.png", None, st)
        if not np.array_equal(np.asarray(img), np.asarray(Image.open(ROOT / item["image"]).convert("RGB"))):
            raise SystemExit(f"replayed render of {item['id']} differs from the saved image; not adding twins")
        checked += 1
    (DATA / "_replay_check.png").unlink(missing_ok=True)
    print(f"replay verified on {checked} harmful images (pixel-identical)")

    twins = []
    for p in pairs:
        h = by_id.get(p["harmful_id"])
        if h is None or h["label"] != "harmful":
            print(f"[skip] {p['harmful_id']} is not a train harmful example")
            continue
        st = styles[h["id"]]
        img_rel = f"images/train/twin_{h['id']}.png"
        render_typographic(p["benign_prompt"], ROOT / img_rel, None, st)
        twins.append(
            dict(
                id=f"twin_{h['id']}",
                label="benign",
                source="minimal_pair",
                instruction=p["benign_prompt"],
                response=p["benign_response"],
                refusal=None,
                image=img_rel,
                carrier=h["carrier"],
                twin_of=h["id"],
            )
        )
    out = DATA / "manifest_train_pairs.json"
    with open(out, "w") as f:
        json.dump(train + twins, f, indent=1)
    print(f"wrote {len(twins)} twins -> {out} (train had {len(train)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-harmful", type=int, default=300)
    ap.add_argument("--n-benign", type=int, default=300)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ultrachat", action="store_true")
    ap.add_argument("--minimal-pairs", action="store_true", help="add reviewed benign twins (Task 6)")
    args = ap.parse_args()
    if args.minimal_pairs:
        add_minimal_pairs(args)
        raise SystemExit
    build(args.n_harmful, args.n_benign, args.eval_frac, args.seed, not args.no_ultrachat)
