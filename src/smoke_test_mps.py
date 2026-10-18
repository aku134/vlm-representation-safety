"""Load Qwen2-VL-2B on MPS and run one typographic image through it."""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2-VL-2B-Instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
model.eval()
cfg = model.config
n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
print(f"model loaded in {time.time() - t0:.0f}s, layers={n_layers}")

manifest = json.load(open(ROOT / "data" / "manifest_eval.json"))
item = next(x for x in manifest if x["label"] == "benign")
image = Image.open(ROOT / item["image"]).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": str(ROOT / item["image"])},
            {"type": "text", "text": item["carrier"]},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(f"generation took {time.time() - t0:.0f}s")
print(f"instruction in image: {item['instruction'][:100]}")
print(f"model reply: {reply[:400]}")

# hidden-state extraction path used by the probe
with torch.no_grad():
    hs = model(**inputs, output_hidden_states=True).hidden_states
print(f"hidden states: {len(hs)} layers x {tuple(hs[0].shape)}")
print("SMOKE_OK")
