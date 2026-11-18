"""Multimodal circuit-breaker (LoRRA / representation rerouting) trainer for Qwen2-VL.

Ports GraySwanAI/circuit-breakers `src/lorra_circuit_breaker.py` from text-only
Llama/Mistral to a vision-language model. Same core recipe:

  * two forward passes per batch: frozen base and LoRA;
  * retain loss      = L2 on BENIGN (+ harmful->refusal) image+text pairs;
  * circuit-breaker  = ReLU(cosine) on HARMFUL completions at target layers,
                       rerouting the harmful representation (not blocking
                       it from entering — short-circuiting the completion).

Fixes vs the reference + vs our first failed run:

  1. Schedule: progress = step / max_steps (not step/300).
  2. Response-token masking on the CB loss.
  3. CB coefficient floor so training does not end with cb_coeff = 0.
  4. Filter CB data to prompts the *base* model actually complies with
     (pushing on refusal representations was why cosine stayed ~1.0).
  5. Cosine logged as mean over layers in [0, 1] (watch it fall).

Runs on CUDA (Colab T4/L4/A100).

Usage:
    python src/train_cb_vlm.py --manifest data/manifest_train.json \\
        --image-root . --output-dir out/qwen2vl-cb
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class Example:
    image_path: str
    carrier: str
    response: str
    example_id: str = ""


REFUSAL_RE = re.compile(
    r"\bI can(?:'|no)t\b|\bI cannot\b|\bI'm sorry\b|\bI am sorry\b|"
    r"\bI'm not able\b|\bI am not able\b|\bI won't\b|\bI will not\b|"
    r"\bcannot assist\b|\bcan't assist\b|\bcannot help\b|\bcan't help\b|"
    r"\bnot appropriate\b|\bI'm unable\b|\bI am unable\b|\bas an AI\b|"
    r"\bI must refuse\b|\bI refuse\b",
    re.IGNORECASE,
)


def looks_like_refusal(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return True
    return bool(REFUSAL_RE.search(t[:500]))


def load_examples(manifest_path: str, image_root: str):
    """Returns (harmful, retain) example lists before complying filter."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    harmful, retain = [], []
    for item in manifest:
        img = str(Path(image_root) / item["image"])
        eid = item.get("id", "")
        if item["label"] == "harmful":
            harmful.append(Example(img, item["carrier"], item["response"], eid))
            if item.get("refusal"):
                retain.append(Example(img, item["carrier"], item["refusal"], eid))
        else:
            retain.append(Example(img, item["carrier"], item["response"], eid))
    return harmful, retain


@torch.no_grad()
def filter_complying_harmful(
    harmful: list[Example],
    model,
    processor,
    device: str,
    cache_path: Path | None,
    max_new_tokens: int = 80,
) -> list[Example]:
    """Keep only harmful examples where the base model complies (not refuses).

    Circuit-breaker loss teacher-forces a harmful completion. If the base
    model already refuses that prompt, its activations are refusal-shaped
    and the CB gradient has nothing useful to push. Cache results to disk.
    """
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        cache = json.load(open(cache_path))
        print(f"loaded complying cache: {cache_path} ({len(cache)} entries)")

    # Use the unwrapped base so we never accidentally generate with LoRA.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.eval()

    kept = []
    for i, ex in enumerate(harmful):
        key = ex.example_id or ex.image_path
        if key in cache:
            verdict = cache[key]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex.image_path},
                        {"type": "text", "text": ex.carrier},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image = Image.open(ex.image_path).convert("RGB")
            enc = processor(text=[text], images=[image], return_tensors="pt")
            enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
            out = base.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            gen = processor.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            verdict = "refusal" if looks_like_refusal(gen) else "comply"
            cache[key] = verdict
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  filter {i + 1}/{len(harmful)}: {verdict} | {gen[:80]!r}", flush=True)

        if verdict == "comply":
            kept.append(ex)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=1)
        print(f"wrote complying cache -> {cache_path}")

    n_ref = sum(1 for v in cache.values() if v == "refusal")
    print(
        f"complying filter: kept {len(kept)}/{len(harmful)} harmful for CB "
        f"(cache refusals among scanned: {n_ref})"
    )
    if not kept:
        raise RuntimeError(
            "No complying harmful examples left for CB training. "
            "Relax the filter or use a weaker base model."
        )
    return kept


class Batcher:
    """Infinite shuffled batches of Examples, tokenized for Qwen2-VL."""

    def __init__(self, examples, processor, batch_size, max_response_chars, device, seed=0):
        self.examples = list(examples)
        self.processor = processor
        self.batch_size = batch_size
        self.max_response_chars = max_response_chars
        self.device = device
        self.rng = random.Random(seed)
        self._queue = []

    def _messages(self, ex: Example, with_response: bool):
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": ex.image_path},
                {"type": "text", "text": ex.carrier},
            ],
        }
        if not with_response:
            return [user]
        response = ex.response[: self.max_response_chars]
        return [user, {"role": "assistant", "content": [{"type": "text", "text": response}]}]

    def next(self):
        if len(self._queue) < self.batch_size:
            pool = list(self.examples)
            self.rng.shuffle(pool)
            self._queue.extend(pool)
        batch = [self._queue.pop() for _ in range(self.batch_size)]

        full_texts, prompt_texts, images = [], [], []
        for ex in batch:
            full = self.processor.apply_chat_template(
                self._messages(ex, True), tokenize=False, add_generation_prompt=False
            )
            prompt = self.processor.apply_chat_template(
                self._messages(ex, False), tokenize=False, add_generation_prompt=True
            )
            full_texts.append(full)
            prompt_texts.append(prompt)
            images.append(Image.open(ex.image_path).convert("RGB"))

        enc = self.processor(
            text=full_texts, images=images, padding=True, return_tensors="pt"
        )
        prompt_enc = self.processor(
            text=prompt_texts, images=images, padding=True, return_tensors="pt"
        )

        # Right padding: response tokens live in [prompt_len, total_len).
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        total_lens = enc["attention_mask"].sum(dim=1)
        response_mask = torch.zeros_like(enc["attention_mask"])
        for i in range(len(batch)):
            response_mask[i, prompt_lens[i] : total_lens[i]] = 1

        enc = {k: v.to(self.device) for k, v in enc.items()}
        enc["response_mask"] = response_mask.to(self.device)
        return enc


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

_SKIP_KEYS = {"response_mask"}


def _model_inputs(enc):
    return {k: v for k, v in enc.items() if k not in _SKIP_KEYS and v is not None}


def hidden_states(model, enc, layers, adapter_off=False, logits_mask=None):
    """Language-model hidden states at `layers`, as a list of (B, L, H).

    One forward with output_hidden_states=True; we index into it (no extra
    passes). adapter_off=True must use PeftModel.disable_adapter(), NOT
    get_base_model(): get_base_model() still returns modules whose Linear layers
    are LoraLinear and keep applying the LoRA delta — so "frozen" == "lora",
    cosine sticks at 1.0, and any retain L2 is mostly LoRA dropout noise.
    """
    kwargs = dict(
        **_model_inputs(enc),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if adapter_off:
        assert hasattr(model, "disable_adapter"), "frozen pass needs a PeftModel"
        with model.disable_adapter():
            out = model(**kwargs)
    else:
        out = model(**kwargs)
    hs = [out.hidden_states[l] for l in layers]
    if logits_mask is None:
        del out
        return hs
    # Keep only logits that predict response tokens: logits[t] predicts token t+1.
    sel = out.logits[:, :-1][logits_mask[:, 1:].bool()]  # (N, V)
    del out
    return hs, sel


def retain_loss_fn(lora_hs, frozen_hs, attention_mask):
    """Masked L2 between LoRA and frozen hidden states, averaged over layers.

    lora_hs / frozen_hs: lists of (B, L, H), one entry per retain layer, so we
    never materialise a stacked (layers, B, L, H) copy. Per layer: mean over
    non-pad tokens of ||h_lora - h_frozen||_2; then mean over layers (Gray Swan
    convention), so the loss scale does not grow with the number of layers.
    """
    mask = attention_mask.unsqueeze(-1).to(lora_hs[0].dtype)
    denom = attention_mask.sum().clamp_min(1).float()
    total = 0.0
    for lh, fh in zip(lora_hs, frozen_hs):
        norms = torch.norm((lh - fh) * mask, dim=-1, p=2, dtype=torch.float)
        total = total + norms.sum() / denom
    return total / len(lora_hs)


def kl_frozen_lora(frozen_logits, lora_logits, chunk: int = 64):
    """Mean over response tokens of KL(frozen || LoRA) on next-token dists.

    Inputs are (N, V) logits already restricted to response positions. Chunked
    over N so float32 log-softmax over the ~150k vocab stays small.
    """
    n = lora_logits.shape[0]
    if n == 0:
        return lora_logits.sum() * 0.0
    total = 0.0
    for i in range(0, n, chunk):
        lp_l = torch.log_softmax(lora_logits[i : i + chunk].float(), dim=-1)
        lp_f = torch.log_softmax(frozen_logits[i : i + chunk].float(), dim=-1)
        total = total + torch.nn.functional.kl_div(
            lp_l, lp_f, log_target=True, reduction="sum"
        )
    return total / n


def parse_retain_layers(spec: str, target_layers: list[int], num_layers: int) -> list[int]:
    """Hidden-state indices for the retain loss.

    hidden_states[0] is the embedding output; hidden_states[i] is the output of
    decoder layer i-1. LoRA sits on decoder layers 0..max(target), whose outputs
    are hidden_states[1 .. max(target)+1].

      targets  -> the CB target layers only (old behaviour)
      all      -> every hidden state a LoRA layer writes: 1..max(target)+1
      every:K  -> every K-th of those, plus the targets
      full     -> every hidden state 1..num_layers (Gray Swan original)
    """
    top = max(target_layers) + 1
    if spec == "targets":
        layers = set(target_layers)
    elif spec == "all":
        layers = set(range(1, top + 1))
    elif spec.startswith("every:"):
        k = int(spec.split(":", 1)[1])
        layers = {i for i in range(1, top + 1) if i % k == 0} | set(target_layers)
    elif spec == "full":
        layers = set(range(1, num_layers + 1))
    else:
        raise ValueError(f"bad --retain-layers {spec!r}")
    return sorted(layers)


def mean_cosine(lora_h, frozen_h, response_mask):
    """Mean cosine over layers and response tokens, in [~-1, 1]."""
    mask = response_mask.unsqueeze(0).unsqueeze(-1)
    lora_n = lora_h / lora_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    froz_n = frozen_h / frozen_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cos = (lora_n * froz_n * mask).sum(dim=-1)  # (layers, B, L)
    n_layers = lora_h.shape[0]
    return cos.sum() / (mask.sum().clamp_min(1) * n_layers)


def circuit_breaker_cosine_loss(lora_h, frozen_h, response_mask):
    """ReLU(cosine) — Gray Swan style. Gradient is ~0 when lora≈frozen (LoRA init)."""
    mask = response_mask.unsqueeze(0).unsqueeze(-1)
    lora_n = lora_h / lora_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    froz_n = frozen_h / frozen_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cos = (lora_n * froz_n * mask).sum(dim=-1)
    n_layers = lora_h.shape[0]
    return torch.relu(cos).sum() / (mask.sum().clamp_min(1) * n_layers)


def circuit_breaker_rmu_loss(lora_h, frozen_h, response_mask, rand_dirs):
    """RMU-style: push LoRA activations toward a fixed random direction.

    Unlike ReLU(cosine), this has a large gradient at LoRA init (when
    lora_h == frozen_h ≠ random target). rand_dirs: (n_layers, hidden).
    """
    mask = response_mask.unsqueeze(0).unsqueeze(-1)  # (1, B, L, 1)
    # Match magnitude of frozen states so we only change direction.
    scale = frozen_h.norm(dim=-1, keepdim=True).detach().clamp_min(1e-6)
    target = rand_dirs[:, None, None, :] * scale  # (layers, B, L, H)
    diff = (lora_h - target) * mask
    return (diff.float().pow(2).sum()) / (mask.sum().clamp_min(1) * lora_h.shape[-1])


def make_random_dirs(n_layers: int, hidden: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v = torch.randn(n_layers, hidden, generator=g)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return v.to(device)


def schedule_coeffs(step: int, max_steps: int, alpha: float, cb_floor: float):
    """Linear handoff retain↑ / cb↓, but CB never drops below alpha * cb_floor."""
    progress = step / max_steps
    retain_coeff = alpha * progress
    cb_coeff = alpha * max(1.0 - progress, cb_floor)
    return retain_coeff, cb_coeff


# --------------------------------------------------------------------------
# reproducibility / run folders
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def prepare_run_dir(args) -> Path:
    """out/runs/<run_name>/ — refuse to reuse a folder that already has results."""
    run_dir = Path(args.runs_dir) / args.run_name
    if run_dir.exists() and any(p.name != "stage.log" for p in run_dir.iterdir()):
        raise SystemExit(
            f"run folder {run_dir} already exists and is not empty; "
            "pick a new --run-name (earlier runs are never overwritten)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    tmp.replace(path)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(args):
    device = "cuda"
    assert torch.cuda.is_available(), "This trainer targets CUDA (run on Colab)."
    set_seed(args.seed)
    run_dir = prepare_run_dir(args)
    write_json(run_dir / "args.json", dict(vars(args), argv=sys.argv))
    print(f"run folder: {run_dir}")

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"compute dtype: {dtype}")

    # Keep vision tokens small for T4 16GB (two forwards + hidden states).
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    print(f"model loaded: {num_layers} text layers")

    if args.target_layers:
        target_layers = [int(x) for x in args.target_layers.split(",")]
    else:
        target_layers = [round(num_layers * 10 / 32), round(num_layers * 20 / 32)]
    print(f"target layers (CB loss): {target_layers}")

    # Filter BEFORE attaching LoRA — uses the clean base model.
    harmful, retain = load_examples(args.manifest, args.image_root)
    print(f"before filter: {len(harmful)} harmful, {len(retain)} retain")
    if args.smoke:
        # Smoke: only scan a slice for the complying filter (cache entries are
        # still per-id and valid, so later full runs reuse them).
        harmful = harmful[:24]
        retain = retain[:24]
    if args.filter_complying:
        cache_path = Path(args.complying_cache) if args.complying_cache else None
        harmful = filter_complying_harmful(
            harmful, model, processor, device, cache_path
        )
    print(f"training with: {len(harmful)} harmful (CB), {len(retain)} retain")
    torch.cuda.empty_cache()
    set_seed(args.seed)  # filter may or may not have run; reseed before LoRA init

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        # No task_type="CAUSAL_LM" — PeftModelForCausalLM double-passes attention_mask.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(max(target_layers) + 1)),
    )
    model = get_peft_model(model, lora_cfg)
    # Tiny noise on lora_B (normally all-zero) so cosine-style losses are not
    # exactly at the zero-gradient point on step 0.
    with torch.no_grad():
        n_bumped = 0
        for name, p in model.named_parameters():
            if "lora_B" in name and p.requires_grad:
                p.add_(torch.randn_like(p) * args.lora_b_init)
                n_bumped += 1
        print(f"lora_B init noise={args.lora_b_init} on {n_bumped} tensors")
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Random directions for RMU CB loss (one unit vector per target layer).
    cfg = model.get_base_model().config
    if getattr(cfg, "text_config", None) is not None:
        hidden = cfg.text_config.hidden_size
    else:
        hidden = cfg.hidden_size
    rand_dirs = make_random_dirs(len(target_layers), hidden, device, seed=args.seed)
    print(f"CB loss mode: {args.cb_loss} | hidden={hidden}")

    cb_batches = Batcher(harmful, processor, args.batch_size, args.max_response_chars, device, seed=args.seed + 1)
    rt_batches = Batcher(retain, processor, args.batch_size, args.max_response_chars, device, seed=args.seed + 2)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)

    retain_layers = parse_retain_layers(args.retain_layers, target_layers, num_layers)
    print(f"retain layers ({args.retain_layers}): {retain_layers}")

    meta = dict(
        run_name=args.run_name,
        args=vars(args),
        target_layers=target_layers,
        retain_layers=retain_layers,
        lora_layers=list(range(max(target_layers) + 1)),
        n_harmful_cb=len(harmful),
        n_retain=len(retain),
        dtype=str(dtype),
        gpu=torch.cuda.get_device_name(0),
    )
    norm_r, norm_c = None, None  # step-1 divisors when --normalize-losses
    log = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.max_steps + 1):
        retain_coeff, cb_coeff = schedule_coeffs(
            step, args.max_steps, args.alpha, args.cb_coeff_floor
        )

        cb_enc = cb_batches.next()
        rt_enc = rt_batches.next()

        model.eval()
        optim.zero_grad(set_to_none=True)
        raw_retain, raw_cb, cos_now, raw_kl = None, None, None, None
        scaled_retain, scaled_cb = None, None

        # ---- retain: frozen then LoRA, backward, free ----
        # total = retain_coeff * (l2 / norm_r + kl_coeff * kl)
        if retain_coeff > 0:
            want_kl = args.kl_coeff > 0
            lmask = rt_enc["response_mask"] if want_kl else None
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                res = hidden_states(model, rt_enc, retain_layers, adapter_off=True, logits_mask=lmask)
                frozen_rt, frozen_logits = res if want_kl else (res, None)
            with torch.autocast("cuda", dtype=dtype):
                res = hidden_states(model, rt_enc, retain_layers, adapter_off=False, logits_mask=lmask)
                lora_rt, lora_logits = res if want_kl else (res, None)
                r_raw = retain_loss_fn(lora_rt, frozen_rt, rt_enc["attention_mask"])
            if args.normalize_losses and norm_r is None:
                norm_r = max(float(r_raw.detach()), args.norm_floor)
                print(f"normalize: retain step-1 raw={float(r_raw.detach()):.4g} -> divisor {norm_r:.4g}")
            r_term = r_raw / (norm_r or 1.0)
            if want_kl:
                kl_t = kl_frozen_lora(frozen_logits, lora_logits)
                raw_kl = float(kl_t.detach())
                r_term = r_term + args.kl_coeff * kl_t
            r_loss = retain_coeff * r_term
            raw_retain = float(r_raw.detach())
            scaled_retain = float(r_loss.detach())
            scaler.scale(r_loss).backward()
            del frozen_rt, lora_rt, r_loss, r_raw, r_term, frozen_logits, lora_logits
            torch.cuda.empty_cache()

        # ---- circuit breaker: frozen then LoRA, backward, free ----
        if cb_coeff > 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                frozen_cb = torch.stack(hidden_states(
                    model, cb_enc, target_layers, adapter_off=True
                ))
            with torch.autocast("cuda", dtype=dtype):
                lora_cb = torch.stack(hidden_states(
                    model, cb_enc, target_layers, adapter_off=False
                ))
                if args.cb_loss == "rmu":
                    raw_cb_t = circuit_breaker_rmu_loss(
                        lora_cb, frozen_cb, cb_enc["response_mask"], rand_dirs
                    )
                else:
                    raw_cb_t = circuit_breaker_cosine_loss(
                        lora_cb, frozen_cb, cb_enc["response_mask"]
                    )
                if args.normalize_losses and norm_c is None:
                    norm_c = max(float(raw_cb_t.detach()), args.norm_floor)
                    print(f"normalize: CB step-1 raw={float(raw_cb_t.detach()):.4g} -> divisor {norm_c:.4g}")
                c_loss = cb_coeff * raw_cb_t / (norm_c or 1.0)
            raw_cb = float(raw_cb_t.detach())
            scaled_cb = float(c_loss.detach())
            with torch.no_grad():
                cos_now = float(mean_cosine(lora_cb, frozen_cb, cb_enc["response_mask"]))
            scaler.scale(c_loss).backward()
            del frozen_cb, lora_cb, c_loss, raw_cb_t
            torch.cuda.empty_cache()

        scaler.unscale_(optim)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        scaler.step(optim)
        scaler.update()

        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            rec = dict(
                step=step,
                retain_coeff=round(retain_coeff, 4),
                cb_coeff=round(cb_coeff, 4),
                retain_loss_raw=raw_retain,
                cb_loss_raw=raw_cb,
                retain_loss_scaled=scaled_retain,
                cb_loss_scaled=scaled_cb,
                cb_cosine=None if cos_now is None else round(cos_now, 4),
                retain_loss_norm=None if raw_retain is None else raw_retain / (norm_r or 1.0),
                cb_loss_norm=None if raw_cb is None else raw_cb / (norm_c or 1.0),
                kl_raw=raw_kl,
                kl_scaled=None if raw_kl is None else retain_coeff * args.kl_coeff * raw_kl,
                norm_retain_div=norm_r,
                norm_cb_div=norm_c,
                grad_norm=round(grad_norm, 4),
                elapsed_s=round(time.time() - t0, 1),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
            )
            log.append(rec)
            print(rec, flush=True)
            # Flush the log every time so a crash/OOM still leaves a record.
            write_json(run_dir / "train_log.json", dict(meta, log=log, status="running"))
            if (
                cos_now is not None
                and step == 50
                and all((e.get("cb_cosine") or 0) > 0.95 for e in log if e.get("cb_cosine") is not None)
            ):
                print(
                    "  [warn] cb_cosine never dropped below 0.95 in first 50 steps — "
                    "reroute may not be taking",
                    flush=True,
                )

        if args.save_every and step % args.save_every == 0 and step < args.max_steps:
            ckpt = run_dir / f"step{step}"
            model.save_pretrained(str(ckpt))
            print(f"checkpoint -> {ckpt}", flush=True)
            torch.cuda.empty_cache()

    model.save_pretrained(str(run_dir / "adapter"))
    processor.save_pretrained(str(run_dir / "adapter"))
    write_json(
        run_dir / "train_log.json",
        dict(meta, log=log, status="done", train_minutes=round((time.time() - t0) / 60, 2)),
    )
    print(f"adapter + log saved to {run_dir}")

    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(run_dir / "merged"))
        processor.save_pretrained(str(run_dir / "merged"))
        print(f"merged model saved to {run_dir / 'merged'}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="results go to <runs-dir>/<run-name>/")
    ap.add_argument("--runs-dir", default="out/runs")
    ap.add_argument("--smoke", action="store_true", help="10 steps, log every step, no checkpoints")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--manifest", default="data/manifest_train.json")
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--target-layers", default="9,18", help="comma list of hidden-state indices")
    ap.add_argument(
        "--retain-layers",
        default="targets",
        help="targets | all | every:K | full  (hidden-state indices for the retain L2)",
    )
    ap.add_argument(
        "--kl-coeff",
        type=float,
        default=0.0,
        help="weight of KL(frozen||LoRA) on retain-batch response tokens (0 = off)",
    )
    ap.add_argument(
        "--normalize-losses",
        action="store_true",
        help="divide retain L2 and CB loss by their step-1 raw values (KL is not normalized)",
    )
    ap.add_argument("--norm-floor", type=float, default=1e-2, help="min divisor for --normalize-losses")
    ap.add_argument("--alpha", type=float, default=10.0, help="lorra_alpha")
    ap.add_argument(
        "--cb-coeff-floor",
        type=float,
        default=0.2,
        help="minimum fraction of alpha kept on CB loss (prevents end-of-run CB=0)",
    )
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1, help="use 1 on T4 16GB")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-b-init", type=float, default=1e-3, help="noise on lora_B to break zero-grad at init")
    ap.add_argument(
        "--cb-loss",
        choices=["rmu", "cosine"],
        default="rmu",
        help="rmu=push toward random dir (has grad at init); cosine=Gray Swan ReLU(cos)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-response-chars", type=int, default=800)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0, help="0 = only save the final adapter")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--merge", action="store_true", help="also save merged full model")
    ap.add_argument(
        "--filter-complying",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only CB-train on prompts the base model complies with (default: on)",
    )
    ap.add_argument(
        "--complying-cache",
        default="out/complying_cache.json",
        help="base-model comply/refusal labels, shared by all runs (same data, greedy decoding)",
    )
    return ap


def apply_smoke(args):
    if args.smoke:
        args.max_steps = 10
        args.log_every = 1
        args.save_every = 0
    return args


if __name__ == "__main__":
    train(apply_smoke(build_argparser().parse_args()))
