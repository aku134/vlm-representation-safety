"""Image perturbations used to stress-test defenses.

Each perturbation simulates an attacker degrading the typographic image to
evade pixel/OCR-level filters while hoping the VLM still reads the
instruction. Severity is per-perturbation, roughly ordered weak -> strong.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr = arr + np.random.default_rng(0).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(degrees, expand=False, fillcolor=(255, 255, 255))


def occlude(img: Image.Image, frac: float, seed: int = 0) -> Image.Image:
    """Cover `frac` of the image area with random gray patches."""
    img = img.copy()
    rng = random.Random(seed)
    w, h = img.size
    covered = 0.0
    arr = np.asarray(img).copy()
    while covered < frac:
        pw, ph = rng.randint(w // 10, w // 4), rng.randint(h // 10, h // 4)
        x, y = rng.randint(0, w - pw), rng.randint(0, h - ph)
        arr[y : y + ph, x : x + pw] = 128
        covered += (pw * ph) / (w * h)
    return Image.fromarray(arr)


def downscale_blur(img: Image.Image, factor: int) -> Image.Image:
    w, h = img.size
    small = img.resize((w // factor, h // factor), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# name -> list of (severity_label, fn)
PERTURBATIONS = {
    "clean": [("none", lambda im: im)],
    "noise": [
        ("sigma=15", lambda im: gaussian_noise(im, 15)),
        ("sigma=35", lambda im: gaussian_noise(im, 35)),
        ("sigma=60", lambda im: gaussian_noise(im, 60)),
    ],
    "jpeg": [
        ("q=50", lambda im: jpeg_compress(im, 50)),
        ("q=25", lambda im: jpeg_compress(im, 25)),
        ("q=10", lambda im: jpeg_compress(im, 10)),
    ],
    "rotate": [
        ("7deg", lambda im: rotate(im, 7)),
        ("15deg", lambda im: rotate(im, 15)),
    ],
    "occlude": [
        ("10%", lambda im: occlude(im, 0.10)),
        ("20%", lambda im: occlude(im, 0.20)),
        ("30%", lambda im: occlude(im, 0.30)),
    ],
    "blur": [
        ("2x", lambda im: downscale_blur(im, 2)),
        ("4x", lambda im: downscale_blur(im, 4)),
    ],
}


def iter_perturbations(include_clean: bool = True):
    for name, variants in PERTURBATIONS.items():
        if name == "clean" and not include_clean:
            continue
        for severity, fn in variants:
            yield name, severity, fn
