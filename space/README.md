---
title: Probe vs OCR Harmfulness
emoji: 🔎
colorFrom: teal
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hardware: t4-small
---

# Probe vs OCR — harmfulness checker

Upload a typographic jailbreak image. Both detectors answer **is this harmful?**

- **Activation probe** — Qwen2-VL hidden states  
- **OCR keyword filter** — text recovered from pixels + harmful lexicon  

GPU (T4) recommended. Research demo, not a production safety filter.

Source: [vlm-representation-safety](https://github.com/aku134/vlm-representation-safety)
