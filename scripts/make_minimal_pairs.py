"""Draft benign minimal-pair twins for the harmful TRAIN prompts (Task 6).

Writes data/minimal_pairs.json = [{harmful_id, harmful_prompt, benign_prompt, benign_response}].
REVIEW THE FILE BY HAND before running `build_dataset.py --minimal-pairs`.
Only train-split harmful prompts are sent; the eval split is untouched.

    pip install anthropic   # credentials from ANTHROPIC_API_KEY or `ant auth login`
    python scripts/make_minimal_pairs.py [--limit 20] [--batch 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "minimal_pairs.json"

INSTRUCTIONS = """You are helping build training data for an AI-safety classifier \
(a "circuit breaker" that must separate harmful requests from benign ones that look similar).

For each harmful request below, write a BENIGN TWIN: a request that keeps the same topic \
words, sentence structure and length as closely as possible, but is clearly safe and \
something a helpful assistant should answer. Example: "How do I make a pipe bomb" -> \
"How do I make a pipe organ sound louder". Then write a short, genuinely helpful answer \
(2-5 sentences) to the benign twin. Never include harmful content.

Return ONLY a JSON array, one object per input, in the same order:
[{"harmful_id": "...", "benign_prompt": "...", "benign_response": "..."}]

Requests:
"""


def ask(client, batch):
    listing = "\n".join(f'{x["id"]}: {x["instruction"][:400]}' for x in batch)
    resp = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": INSTRUCTIONS + listing}],
    )
    if resp.stop_reason == "refusal":
        print(f"[refused] batch starting {batch[0]['id']}; skipped")
        return []
    text = "".join(b.text for b in resp.content if b.type == "text")
    start, end = text.find("["), text.rfind("]")
    rows = json.loads(text[start : end + 1])
    want = {x["id"] for x in batch}
    return [r for r in rows if r.get("harmful_id") in want and r.get("benign_prompt")]


def main(args):
    manifest = json.load(open(ROOT / "data" / "manifest_train.json"))
    harmful = [x for x in manifest if x["label"] == "harmful"]
    if args.limit:
        harmful = harmful[: args.limit]
    done = json.load(open(OUT)) if OUT.exists() else []
    have = {r["harmful_id"] for r in done}
    todo = [x for x in harmful if x["id"] not in have]
    by_id = {x["id"]: x for x in harmful}
    client = anthropic.Anthropic()
    for i in range(0, len(todo), args.batch):
        for r in ask(client, todo[i : i + args.batch]):
            r["harmful_prompt"] = by_id[r["harmful_id"]]["instruction"]
            done.append(r)
        OUT.write_text(json.dumps(done, indent=1))
        print(f"{len(done)} pairs so far")
    print(f"wrote {OUT}. Review it before using --minimal-pairs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all train harmful prompts")
    ap.add_argument("--batch", type=int, default=10)
    main(ap.parse_args())
