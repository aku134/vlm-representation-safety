# VLM Representation Safety

Representation-level harmfulness detection for vision-language models under typographic jailbreaks.

**Main finding:** a probe on VLM activations keeps detecting harmful image instructions under occlusion and heavy blur; an OCR keyword filter does not.

**Live demo:** [aku134.github.io/vlm-representation-safety](https://aku134.github.io/vlm-representation-safety/) — static perturbation explorer (no GPU; precomputed rates).

**Interactive tool (GPU):** upload an image and compare probe vs OCR harmfulness decisions:

```bash
# from the project root (CUDA / Colab T4 recommended)
python tool_probe_ocr.py
```

Hugging Face Space scaffold is under `space/` (Gradio + T4). Push that folder to a new Space to host a public checker.

---

## Motivation

Gray Swan’s [circuit-breakers](https://github.com/GraySwanAI/circuit-breakers) ([Zou et al. 2024](https://arxiv.org/abs/2406.04313)) place refusal in model *representations* rather than input filters. The public trainer is text-only. A simple multimodal bypass is to render the harmful instruction into an image and send a benign carrier prompt such as “follow the instructions in the image.”

This project asks whether safety that lives in activations survives that attack and image degradation better than safety that lives on the input (OCR).

---

## Setup

| Item | Value |
|---|---|
| Model | [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Attack | Typographic jailbreak — harmful text rendered into 448×448 images |
| Data | Gray Swan circuit-breaker prompts + XSTest benign prompts |
| Train / probe | Colab T4 via `train_colab.ipynb` |
| OCR baseline | macOS Vision (`src/ocr_baseline.py`) |

---

## Method

**Activation probe.** A small classifier (logistic regression / MLP) trained on the VLM’s *hidden states*, not on OCR text. Harmful and benign image+prompt pairs produce different activation patterns; the probe learns that difference as a **harmfulness signal**. At test time it scores layer-22 activations (max over tokens); above a threshold tuned for &lt;1% benign false positives, it fires.

**Circuit breaker.** A LoRA adapter trained with Gray Swan’s LoRRA recipe so that when mid-layer activations look harmful, generation is **rerouted** away from continuing a harmful completion (refusal or incoherent collapse), while a retain loss keeps answers close to the base model on benign traffic. The harmful image+prompt still enters the model; the break is in representation space, not an input filter.

1. Build typographic jailbreak images and train/eval manifests (`src/build_dataset.py`).
2. Extract activations and train the harmfulness probe (`src/extract_activations.py`, `src/train_probe.py`) — RepE-style, max-over-tokens, layer chosen by held-out AUC.
3. Evaluate the probe against an OCR keyword baseline under noise, JPEG, rotation, occlusion, and blur (`src/perturb_eval.py`).
4. Port a multimodal circuit-breaker LoRA (`src/train_cb_vlm.py`) and compare base vs adapter generations (`src/eval_cb.py`).

---

## Results

### Probe vs OCR (primary)

Held-out harmful images (**n=30**). Layer-22 activation probe vs OCR. Numbers from [`results/outputs/perturb_results.json`](results/outputs/perturb_results.json); chart in [`results/outputs/perturb_chart.png`](results/outputs/perturb_chart.png).

| Perturbation | Probe | OCR |
|---|---|---|
| Clean | 97% | 100% |
| Noise / JPEG | ~97% | ~97–100% |
| Occlude 10% | **73%** | **40%** |
| Occlude 20% | **73%** | **37%** |
| Occlude 30% | **83%** | **30%** |
| Blur 4× | **97%** | **40%** |

On clean and lightly degraded images both methods look strong. Under occlusion and heavy blur, OCR detection collapses while the probe continues to fire.

**Takeaway:** representation-level detection tracks what the VLM understood; input-level OCR fails open under the same degradations.

### Circuit breaker (secondary)

Goal of a circuit breaker: if the model is about to continue a harmful answer, reroute those activations so generation collapses, without breaking benign answers. A LoRA circuit breaker was trained and evaluated on the same typographic setup. On clean harmful images (n=20), the base model complied **30%** of the time; the circuit-broken model complied **50%** — no behavioral safety gain in this run. Benign clean compliance stayed at 100% for both. See [`results/outputs/eval_summary.json`](results/outputs/eval_summary.json).

The circuit-breaker port is included for completeness; it did not produce a solid generation-level win. The supported claim from this repo is the probe-vs-OCR result.

---

## Repository layout

```
src/
  build_dataset.py      # typographic images + manifests
  train_cb_vlm.py       # multimodal LoRRA / circuit-breaker trainer
  eval_cb.py            # base vs circuit-broken generations
  extract_activations.py
  train_probe.py
  perturb_eval.py       # probe vs OCR under degradations
  ocr_baseline.py
  pack_for_colab.py
train_colab.ipynb       # GPU pipeline (train, probe, eval)
results/outputs/        # eval summaries, perturb rates, chart, train log
```

---

## Running 

**Probe vs OCR tool (GPU):**

```bash 
pip install -U transformers accelerate qwen-vl-utils pillow scikit-learn joblib gradio
# needs results/outputs/probe.joblib (or outputs/probe.joblib)
python tool_probe_ocr.py
```

Upload a typographic image (or pick a held-out example), optionally perturb it, and see:
- activation probe → HARMFUL / not harmful  
- OCR keyword filter → HARMFUL / not harmful  

**Colab (GPU):** open `train_colab.ipynb`, select a T4 runtime, upload a `colab_upload.zip` produced by `src/pack_for_colab.py`, and run the cells.

**Local (dataset / OCR, no CUDA required for those steps):**

```bash
pip install -U transformers accelerate peft qwen-vl-utils pillow scikit-learn joblib matplotlib
python src/build_dataset.py
python src/ocr_baseline.py 30
```

Full VLM train and activation extraction need a CUDA GPU.

---

## License

MIT — see [LICENSE](LICENSE).
