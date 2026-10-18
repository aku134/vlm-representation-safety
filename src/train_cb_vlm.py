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
import random
import re
from dataclasses import dataclass
from pathlib import Path

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


def hidden_states(model, enc, layers=None, adapter_off=False):
    """Language-model hidden states.

    adapter_off=True must use PeftModel.disable_adapter(), NOT get_base_model().
    get_base_model() still returns modules whose Linear layers are LoraLinear and
    keep applying the LoRA delta — so "frozen" == "lora", cosine sticks at 1.0,
    and any retain L2 is mostly train-mode LoRA dropout noise.
    """
    kwargs = dict(
        **_model_inputs(enc),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if adapter_off and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            out = model(**kwargs)
    else:
        out = model(**kwargs)
    hs = out.hidden_states
    if layers is None:
        return torch.stack(hs)
    return torch.stack([hs[l] for l in layers])


def retain_loss_fn(lora_h, frozen_h, attention_mask):
    """Masked L2 between LoRA and frozen hidden states, all layers."""
    mask = attention_mask.unsqueeze(0).unsqueeze(-1)
    diff = (lora_h - frozen_h) * mask
    norms = torch.norm(diff, dim=-1, p=2, dtype=torch.float)
    return norms.sum() / mask.sum().clamp_min(1.0)


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
# training
# --------------------------------------------------------------------------

def train(args):
    device = "cuda"
    assert torch.cuda.is_available(), "This trainer targets CUDA (run on Colab)."
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
    if args.filter_complying:
        cache_path = Path(args.complying_cache) if args.complying_cache else None
        harmful = filter_complying_harmful(
            harmful, model, processor, device, cache_path
        )
    print(f"training with: {len(harmful)} harmful (CB), {len(retain)} retain")
    torch.cuda.empty_cache()

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

    cb_batches = Batcher(harmful, processor, args.batch_size, args.max_response_chars, device, seed=1)
    rt_batches = Batcher(retain, processor, args.batch_size, args.max_response_chars, device, seed=2)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)

    # Retain only at CB target layers (not all 29) — otherwise T4 OOMs holding
    # two full hidden-state stacks.
    retain_layers = list(target_layers)
    print(f"retain layers: {retain_layers} (memory-safe; was all layers)")

    log = []
    for step in range(1, args.max_steps + 1):
        retain_coeff, cb_coeff = schedule_coeffs(
            step, args.max_steps, args.alpha, args.cb_coeff_floor
        )

        cb_enc = cb_batches.next()
        rt_enc = rt_batches.next()

        model.eval()
        optim.zero_grad(set_to_none=True)
        r_loss_val, c_loss_val, cos_now = 0.0, 0.0, None

        # ---- retain: frozen then LoRA, backward, free ----
        if retain_coeff > 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                frozen_rt = hidden_states(
                    model, rt_enc, retain_layers, adapter_off=True
                )
            with torch.autocast("cuda", dtype=dtype):
                lora_rt = hidden_states(
                    model, rt_enc, retain_layers, adapter_off=False
                )
                r_loss = retain_coeff * retain_loss_fn(
                    lora_rt, frozen_rt, rt_enc["attention_mask"]
                )
            r_loss_val = float(r_loss.detach() / retain_coeff)
            scaler.scale(r_loss).backward()
            del frozen_rt, lora_rt, r_loss
            torch.cuda.empty_cache()

        # ---- circuit breaker: frozen then LoRA, backward, free ----
        if cb_coeff > 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                frozen_cb = hidden_states(
                    model, cb_enc, target_layers, adapter_off=True
                )
            with torch.autocast("cuda", dtype=dtype):
                lora_cb = hidden_states(
                    model, cb_enc, target_layers, adapter_off=False
                )
                if args.cb_loss == "rmu":
                    raw_cb = circuit_breaker_rmu_loss(
                        lora_cb, frozen_cb, cb_enc["response_mask"], rand_dirs
                    )
                else:
                    raw_cb = circuit_breaker_cosine_loss(
                        lora_cb, frozen_cb, cb_enc["response_mask"]
                    )
                c_loss = cb_coeff * raw_cb
            c_loss_val = float(raw_cb.detach())
            with torch.no_grad():
                cos_now = float(mean_cosine(lora_cb, frozen_cb, cb_enc["response_mask"]))
            scaler.scale(c_loss).backward()
            del frozen_cb, lora_cb, c_loss, raw_cb
            torch.cuda.empty_cache()

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optim)
        scaler.update()

        if step % args.log_every == 0 or step == 1:
            rec = dict(
                step=step,
                retain_coeff=round(retain_coeff, 3),
                cb_coeff=round(cb_coeff, 3),
                retain_loss=r_loss_val,
                cb_loss=c_loss_val,
                cb_cosine=None if cos_now is None else round(cos_now, 4),
            )
            log.append(rec)
            print(rec, flush=True)
            # Only warn once at the checkpoint step if the *run so far* never
            # left the stuck-at-1 regime (batch_size=1 makes single spikes normal).
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

        if args.save_every and step % args.save_every == 0:
            ckpt = Path(args.output_dir) / f"step{step}"
            model.save_pretrained(str(ckpt))
            print(f"checkpoint -> {ckpt}", flush=True)
            torch.cuda.empty_cache()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "adapter"))
    processor.save_pretrained(str(out / "adapter"))
    with open(out / "train_log.json", "w") as f:
        json.dump(
            dict(
                args=vars(args),
                target_layers=target_layers,
                n_harmful_cb=len(harmful),
                n_retain=len(retain),
                log=log,
            ),
            f,
            indent=1,
        )
    print(f"adapter + log saved to {out}")

    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out / "merged"))
        processor.save_pretrained(str(out / "merged"))
        print(f"merged model saved to {out / 'merged'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--manifest", default="data/manifest_train.json")
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--output-dir", default="out/qwen2vl-cb")
    ap.add_argument("--target-layers", default="", help="comma list; default = depth-scaled 10,20")
    ap.add_argument("--alpha", type=float, default=10.0, help="lorra_alpha")
    ap.add_argument(
        "--cb-coeff-floor",
        type=float,
        default=0.2,
        help="minimum fraction of alpha kept on CB loss (prevents end-of-run CB=0)",
    )
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1, help="use 1 on T4 16GB")
    ap.add_argument("--lr", type=float, default=2e-4)
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
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--merge", action="store_true", help="also save merged full model")
    ap.add_argument(
        "--filter-complying",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only CB-train on prompts the base model complies with (default: on)",
    )
    ap.add_argument(
        "--complying-cache",
        default="out/qwen2vl-cb/complying_cache.json",
        help="cache base-model comply/refusal labels so filter is not re-run",
    )
    args = ap.parse_args()
    train(args)
