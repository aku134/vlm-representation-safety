#!/usr/bin/env bash
# Train -> generate -> judge for one run. On any failure writes out/runs/<name>/FAILED.txt
# with the reason and exits 0, so a sweep continues (the plan: log failures, keep going).
#   bash scripts/run_experiment.sh R0 --retain-layers targets
#   bash scripts/run_experiment.sh smoke --smoke --retain-layers all     (smoke: 10 steps, 4 eval items)
set -uo pipefail
NAME="$1"; shift
RUN="out/runs/$NAME"
SMOKE=""; for a in "$@"; do [[ "$a" == "--smoke" ]] && SMOKE="--smoke"; done
BASE_FROM="out/runs/base"; [[ -n "$SMOKE" ]] && BASE_FROM="out/runs/base_smoke"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

fail() { mkdir -p "$RUN"; { echo "$1"; tail -n 20 "$RUN/stage.log" 2>/dev/null; } > "$RUN/FAILED.txt"; echo "FAILED: $1"; exit 0; }
mkdir -p "$RUN"
python src/train_cb_vlm.py --run-name "$NAME" "$@" 2>&1 | tee "$RUN/stage.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "training failed ($(grep -m1 -oE 'OutOfMemoryError|CUDA out of memory|Error[^ ]*' "$RUN/stage.log" || echo see log))"
python src/eval_cb.py --run-dir "$RUN" --base-from "$BASE_FROM" $SMOKE 2>&1 | tee "$RUN/stage.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "generation eval failed"
python src/judge.py --run-dir "$RUN" 2>&1 | tee "$RUN/stage.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "judge failed"
rm -f "$RUN/stage.log"
echo "done: $RUN"
