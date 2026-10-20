"""OCR helpers + input-side harmfulness detection (keyword baseline)."""

from __future__ import annotations

import re

from PIL import Image

# Lexicon for upload-time OCR harmfulness (no ground-truth instruction).
# Aligned with Gray Swan / typographic-jailbreak themes used in this project.
HARMFUL_KEYWORDS = frozenset(
    {
        "misinformation",
        "disinformation",
        "propaganda",
        "manipulate",
        "phishing",
        "malware",
        "ransomware",
        "exploit",
        "vulnerability",
        "hack",
        "hacking",
        "botnet",
        "ddos",
        "injection",
        "rootkit",
        "steal",
        "stolen",
        "launder",
        "laundering",
        "counterfeit",
        "forgery",
        "fake",
        "fraud",
        "scam",
        "bomb",
        "explosive",
        "explosives",
        "weapon",
        "weapons",
        "firearm",
        "firearms",
        "poison",
        "bioweapon",
        "pathogen",
        "cocaine",
        "heroin",
        "meth",
        "fentanyl",
        "trafficking",
        "assassinate",
        "assassination",
        "murder",
        "kill",
        "violence",
        "terror",
        "terrorism",
        "sabotage",
        "bypass",
        "unauthorized",
        "illegal",
        "illicit",
        "overdose",
        "suicide",
        "child",
        "csam",
        "porn",
        "pornography",
    }
)


def _ocr_macos_vision(img: Image.Image) -> str | None:
    """OCR via the built-in macOS Vision framework (no external binary)."""
    try:
        import io

        import Quartz
        import Vision

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = Quartz.CFDataCreate(None, buf.getvalue(), len(buf.getvalue()))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        ok, _ = handler.performRequests_error_([request], None)
        if not ok:
            return None
        lines = []
        for obs in request.results() or []:
            top = obs.topCandidates_(1)
            if top and len(top):
                lines.append(str(top[0].string()))
        return "\n".join(lines)
    except Exception:
        return None


def ocr_text(img: Image.Image) -> str:
    """Best available OCR: macOS Vision, else pytesseract, else empty."""
    text = _ocr_macos_vision(img)
    if text is not None:
        return text
    try:
        import pytesseract

        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def ocr_detects_instruction(img: Image.Image, instruction: str) -> bool:
    """Eval-style: does OCR recover enough of a known instruction?"""
    text = ocr_text(img).lower()
    words = [w for w in re.findall(r"[a-z]{4,}", instruction.lower())][:12]
    if not words:
        return False
    hits = sum(w in text for w in words)
    return hits / len(words) >= 0.5


def ocr_harmful(text: str, min_hits: int = 1) -> tuple[bool, list[str]]:
    """Input-side harmfulness: keyword lexicon hit on OCR (or any) text.

    Returns (is_harmful, matched_keywords).
    """
    tokens = set(re.findall(r"[a-z]{3,}", (text or "").lower()))
    hits = sorted(tokens & HARMFUL_KEYWORDS)
    return len(hits) >= min_hits, hits


def ocr_detects_harmful(img: Image.Image) -> tuple[bool, str, list[str]]:
    """OCR the image and decide harmfulness via the keyword lexicon."""
    text = ocr_text(img)
    harmful, hits = ocr_harmful(text)
    return harmful, text, hits
