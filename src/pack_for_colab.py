"""Build one zip to upload to Colab: images, manifests, and the Python sources."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "colab_upload.zip"

INCLUDE = [
    ROOT / "data" / "manifest_train.json",
    ROOT / "data" / "manifest_eval.json",
    ROOT / "outputs" / "ocr_results.json",
]
SRC_FILES = [
    "train_cb_vlm.py",
    "vlm_common.py",
    "perturbations.py",
    "extract_activations.py",
    "train_probe.py",
    "eval_cb.py",
    "perturb_eval.py",
]


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            if p.exists():
                z.write(p, p.relative_to(ROOT))
        for name in SRC_FILES:
            z.write(ROOT / "src" / name, f"src/{name}")
        for img in (ROOT / "images").rglob("*.png"):
            z.write(img, img.relative_to(ROOT))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print("Upload this single file to Colab and unzip it.")


if __name__ == "__main__":
    main()
