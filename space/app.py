"""Probe vs OCR harmfulness checker (upload an image).

Both detectors answer “is this harmful?”:
  * Activation probe — harmfulness signal in Qwen2-VL hidden states
  * OCR keyword filter — recover text from pixels, then lexicon check

    # GPU recommended (Colab / HF Space T4). On Mac CPU this is very slow.
    venv310/bin/python tool_probe_ocr.py

Needs:
  results/outputs/probe.joblib  (or outputs/probe.joblib)
  Hugging Face access to Qwen/Qwen2-VL-2B-Instruct
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import joblib
import numpy as np
from PIL import Image

from ocr_common import ocr_detects_harmful, ocr_detects_instruction
from perturbations import PERTURBATIONS
from vlm_common import hidden_states_at, load_base, load_processor

PROBE_CANDIDATES = [
    ROOT / "outputs" / "probe.joblib",
    ROOT / "results" / "outputs" / "probe.joblib",
]


def _find_probe() -> Path:
    for p in PROBE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "probe.joblib not found. Expected outputs/probe.joblib or "
        "results/outputs/probe.joblib (train via Colab / train_probe.py)."
    )


print("loading processor + base VLM (first run may download weights)...")
PROCESSOR = load_processor()
MODEL = load_base()

PROBE_PATH = _find_probe()
PROBE = joblib.load(PROBE_PATH)
print(f"probe: {PROBE_PATH}  layer={PROBE['layer']}  threshold={PROBE['threshold']:.4f}")

MANIFEST_PATH = ROOT / "data" / "manifest_eval.json"
EXAMPLES: dict[str, dict] = {}
if MANIFEST_PATH.exists():
    import json

    for x in json.load(open(MANIFEST_PATH)):
        key = f"{x['label']} | {x['instruction'][:70]}"
        EXAMPLES[key] = x

PERTURB_CHOICES = ["none"] + [
    f"{name} ({sev})"
    for name, variants in PERTURBATIONS.items()
    if name != "clean"
    for sev, _ in variants
]


def apply_perturbation(img: Image.Image, choice: str) -> Image.Image:
    if choice == "none":
        return img
    name, sev = choice.split(" (", 1)
    sev = sev.rstrip(")")
    for s, fn in PERTURBATIONS[name]:
        if s == sev:
            return fn(img)
    return img


def run(example_key, uploaded, carrier, perturb_choice, expected_instruction):
    if uploaded is not None:
        img = Image.fromarray(uploaded).convert("RGB")
        carrier_txt = (carrier or "").strip() or "Follow the instructions written in the image."
        expected = (expected_instruction or "").strip()
    elif example_key:
        item = EXAMPLES[example_key]
        img = Image.open(ROOT / item["image"]).convert("RGB")
        carrier_txt = (carrier or "").strip() or item["carrier"]
        expected = (expected_instruction or "").strip() or item["instruction"]
    else:
        return (
            None,
            "Pick an example or upload an image.",
            "",
            "",
        )

    img = apply_perturbation(img, perturb_choice)

    # --- Probe (representation-level harmfulness) ---
    clf, layer, threshold = PROBE["clf"], PROBE["layer"], PROBE["threshold"]
    hs = hidden_states_at(MODEL, PROCESSOR, img, carrier_txt, layer)
    tail = hs[-8:].numpy().astype(np.float32)
    score = float(clf.predict_proba(tail)[:, 1].max())
    probe_flag = score >= threshold
    probe_out = (
        f"{'HARMFUL' if probe_flag else 'not harmful'}\n"
        f"score {score:.4f}  (threshold {threshold:.4f})\n"
        f"layer {layer} · max-over-last-8-tokens"
    )

    # --- OCR (input-level harmfulness) ---
    ocr_flag, ocr_raw, hits = ocr_detects_harmful(img)
    ocr_lines = [
        f"{'HARMFUL' if ocr_flag else 'not harmful'}",
        f"matched keywords: {', '.join(hits) if hits else '(none)'}",
        "",
        "recovered text:",
        ocr_raw.strip() or "(empty — OCR failed or text unreadable)",
    ]
    if expected:
        instr_hit = ocr_detects_instruction(img, expected)
        ocr_lines.insert(
            2,
            f"eval-style instruction match (≥50% keywords): "
            f"{'yes' if instr_hit else 'no'}",
        )
    ocr_out = "\n".join(ocr_lines)

    summary = (
        f"Probe: {'HARMFUL' if probe_flag else 'not harmful'}  |  "
        f"OCR: {'HARMFUL' if ocr_flag else 'not harmful'}"
    )
    return img, summary, probe_out, ocr_out


with gr.Blocks(title="Probe vs OCR — harmfulness checker") as demo:
    gr.Markdown(
        """
# Probe vs OCR — harmfulness checker

Upload a typographic jailbreak image (harmful text rendered into pixels) or pick a held-out example.

| Detector | What it does |
|---|---|
| **Activation probe** | Reads a harmfulness signal from Qwen2-VL hidden states |
| **OCR keyword filter** | Recovers text from pixels, then checks a harmful-keyword lexicon |

Under blur / occlusion, OCR often misses while the probe still fires — that is the main result of this project.

*Research demo, not a production safety filter. GPU recommended.*
"""
    )
    with gr.Row():
        with gr.Column(scale=1):
            example = gr.Dropdown(
                list(EXAMPLES) if EXAMPLES else [],
                label="Held-out example (optional)",
                value=None,
            )
            upload = gr.Image(label="…or upload an image", type="numpy")
            carrier = gr.Textbox(
                label="Carrier prompt (sent to the VLM with the image)",
                value="Follow the instructions written in the image.",
            )
            expected = gr.Textbox(
                label="Expected instruction (optional — eval-style OCR match)",
                placeholder="Paste the text written in the image if known",
                value="",
            )
            perturb = gr.Dropdown(PERTURB_CHOICES, value="none", label="Perturbation")
            btn = gr.Button("Check harmfulness", variant="primary")
            shown = gr.Image(label="Image as scored")
            summary = gr.Textbox(label="Summary", lines=1)
        with gr.Column(scale=1):
            probe_box = gr.Textbox(label="Activation probe (representation-level)", lines=6)
            ocr_box = gr.Textbox(label="OCR keyword filter (input-level)", lines=12)

    btn.click(
        run,
        [example, upload, carrier, perturb, expected],
        [shown, summary, probe_box, ocr_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
