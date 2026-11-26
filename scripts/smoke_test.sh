#!/bin/bash
#
# GPU smoke test: short end-to-end runs of both pipelines on real data and
# real timm models, checking the behaviour the unit tests can't reach (CUDA,
# AMP, multi-GPU launch, file locks, resume, seeding, center readout).
#
# Usage (from the repo root):
#   ./scripts/smoke_test.sh --data pathfinder=/data/pathfinder --data tilt=/data/tilt --gpus 2
#
# Each --data TASK=PATH adds a task; at least one is required. Pretrained
# weights are downloaded by timm on first use. Exits non-zero if any check fails.
#

set -u

GPU_COUNT=2
OUTPUT_DIR="results/smoke_$(date +%Y%m%d_%H%M%S)"
MODELS=(resnet18 vit_small_patch16_224)  # one CNN (spatial center), one ViT (center token)
TASKS=()
declare -A DATA_DIRS

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            TASKS+=("${2%%=*}"); DATA_DIRS["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --gpus)
            GPU_COUNT="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,13p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done
[[ ${#TASKS[@]} -gt 0 ]] || { echo "Error: pass at least one --data TASK=PATH"; exit 1; }

mkdir -p "$OUTPUT_DIR"
MODELS_LIST="$OUTPUT_DIR/models.txt"
printf '%s\n' "${MODELS[@]}" > "$MODELS_LIST"

FAILURES=0
pass() { echo "  PASS: $*"; }
fail() { echo "  FAIL: $*"; FAILURES=$((FAILURES + 1)); }
section() { echo ""; echo "=== $* ==="; }

# Python helper: print a model's best_val_acc from a JSON log ("missing" if absent)
best_val() {
    python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1]))[sys.argv[2]]['best_val_acc'])
except Exception:
    print('missing')" "$1" "$2"
}

# Check one pipeline run: every model has a result, no errors, reload checks passed
check_run() {  # check_run <label> <log_json> <errors_json> <gpu_log_glob>
    local label=$1 log=$2 errs=$3 glob=$4
    for m in "${MODELS[@]}"; do
        local v; v=$(best_val "$log" "$m")
        if [[ "$v" == "missing" ]]; then fail "$label: no result for $m"; else pass "$label: $m best_val=$v"; fi
    done
    if [[ -s "$errs" ]] && [[ "$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$errs")" != "0" ]]; then
        fail "$label: errors recorded in $errs:"; cat "$errs"
    fi
    local n; n=$(cat $glob 2>/dev/null | grep -c "Reload check passed" || true)
    if [[ "$n" -ge ${#MODELS[@]} ]]; then pass "$label: $n reload checks passed"; else fail "$label: only $n reload checks passed"; fi
    if grep -l "RELOAD CHECK FAILED\|Traceback" $glob >/dev/null 2>&1; then
        fail "$label: tracebacks or reload failures in $(grep -l "RELOAD CHECK FAILED\|Traceback" $glob | tr '\n' ' ')"
    fi
}

# =============================================================================
section "0. Environment"
python3 -c "
import torch, timm
print(f'  torch {torch.__version__}, timm {timm.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
" || { echo "  FAIL: torch/timm not importable (pip install -e .)"; exit 1; }
[[ "$(python3 -c 'import torch; print(torch.cuda.device_count())')" -ge "$GPU_COUNT" ]] \
    || fail "fewer than $GPU_COUNT GPUs visible"

# =============================================================================
section "1. Center readout on real timm models"
python3 - <<'EOF' && pass "center readout specs match expectations" || fail "center readout specs (see above)"
import timm, torch
from lra_simple.common import center_readout_spec, describe_center_readout
from lra_simple.common import get_model_config

expected = {
    'vit_small_patch16_224': {'kind': 'token', 'index': 1 + 7 * 14 + 7, 'grid': (14, 14)},
    # resnet18's default weights have test_input_size 288 (preferred by get_model_config): 288/32 = 9
    'resnet18': {'kind': 'spatial', 'index': (4, 4), 'grid': (9, 9)},
    'swin_tiny_patch4_window7_224': ValueError,
}
ok = True
for name, want in expected.items():
    backbone = timm.create_model(name, pretrained=False, num_classes=0)
    try:
        spec, dim = center_readout_spec(backbone, get_model_config(name)['input_size'])
        got = f"{describe_center_readout(spec)}, dim {dim}"
        good = spec == want
    except ValueError as e:
        got, good = f"ValueError: {e}", want is ValueError
    print(f"    {'ok ' if good else 'BAD'} {name}: {got}")
    ok &= good
raise SystemExit(0 if ok else 1)
EOF

# =============================================================================
section "2. Linear probe, multi-GPU launch (${MODELS[*]} on $GPU_COUNT GPUs, 2 epochs)"
for task in "${TASKS[@]}"; do
    out="$OUTPUT_DIR/lp"
    ./scripts/launch_linear_probe.sh --task "$task" --data_dir "${DATA_DIRS[$task]}" \
        --models_list "$MODELS_LIST" --gpus "$GPU_COUNT" --epochs 2 --output_dir "$out" \
        > "$OUTPUT_DIR/lp_${task}_launcher.log" 2>&1
    check_run "probe/$task" "$out/${task}_linear_probe.json" "$out/${task}_linear_probe_errors.json" \
        "$out/logs/${task}_lp_gpu*.log"
done

# =============================================================================
section "3. Resume: rerunning the probe into the same output dir skips finished models"
task=${TASKS[0]}
out="$OUTPUT_DIR/lp"
./scripts/launch_linear_probe.sh --task "$task" --data_dir "${DATA_DIRS[$task]}" \
    --models_list "$MODELS_LIST" --gpus "$GPU_COUNT" --epochs 2 --output_dir "$out" \
    > "$OUTPUT_DIR/lp_${task}_resume_launcher.log" 2>&1
n=$(cat "$out"/logs/${task}_lp_gpu*.log | grep -c "SKIP:" || true)
[[ "$n" -eq ${#MODELS[@]} ]] && pass "resume: all ${#MODELS[@]} models skipped" \
    || fail "resume: expected ${#MODELS[@]} SKIP lines, got $n"

# =============================================================================
section "4. Fine-tune, multi-GPU launch (--reduced_grid, 1 epoch)"
# One classification task (the first non-tilt one given) plus tilt if given
FT_TASKS=()
for task in "${TASKS[@]}"; do [[ "$task" != tilt ]] && { FT_TASKS+=("$task"); break; }; done
[[ -n "${DATA_DIRS[tilt]:-}" ]] && FT_TASKS+=(tilt)
for task in "${FT_TASKS[@]}"; do
    out="$OUTPUT_DIR/ft"
    ./scripts/launch_finetune.sh --task "$task" --data_dir "${DATA_DIRS[$task]}" \
        --models_list "$MODELS_LIST" --gpus "$GPU_COUNT" --epochs 1 --reduced_grid \
        --output_dir "$out" > "$OUTPUT_DIR/ft_${task}_launcher.log" 2>&1
    check_run "finetune/$task" "$out/${task}_finetune.json" "$out/${task}_finetune_errors.json" \
        "$out/logs/${task}_ft_gpu*.log"
    lines=$(cat "$out/${task}_finetune_configs.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    [[ "$lines" -eq $(( ${#MODELS[@]} * 6 )) ]] && pass "finetune/$task: $lines per-config JSONL lines" \
        || fail "finetune/$task: expected $(( ${#MODELS[@]} * 6 )) per-config lines, got $lines"
done

# =============================================================================
section "5. Reproducibility: same seed twice (single model, single GPU)"
task=${TASKS[0]}
for run in a b; do
    python3 -m lra_simple.unified_linear_probe --task "$task" --data_dir "${DATA_DIRS[$task]}" \
        --model_name resnet18 --epochs 2 --seed 0 --output_dir "$OUTPUT_DIR/seed_lp_$run" \
        > "$OUTPUT_DIR/seed_lp_$run.log" 2>&1
    python3 -m lra_simple.unified_finetune --task "$task" --data_dir "${DATA_DIRS[$task]}" \
        --model_name resnet18 --epochs 1 --reduced_grid --seed 0 --output_dir "$OUTPUT_DIR/seed_ft_$run" \
        > "$OUTPUT_DIR/seed_ft_$run.log" 2>&1
done
for kind in lp ft; do
    suffix=$([[ $kind == lp ]] && echo linear_probe || echo finetune)
    a=$(best_val "$OUTPUT_DIR/seed_${kind}_a/${task}_${suffix}.json" resnet18)
    b=$(best_val "$OUTPUT_DIR/seed_${kind}_b/${task}_${suffix}.json" resnet18)
    if [[ "$a" == "missing" || "$b" == "missing" ]]; then
        fail "seed/$kind: a run produced no result (see $OUTPUT_DIR/seed_${kind}_*.log)"
    elif [[ "$a" == "$b" ]]; then
        pass "seed/$kind: identical best_val ($a)"
    else
        # Not a failure: cuDNN/AMP kernels aren't bitwise deterministic. Large gaps are worth a look.
        echo "  NOTE: seed/$kind: best_val differs between repeats: $a vs $b"
    fi
done

# =============================================================================
section "6. Leftovers"
tmps=$(find "$OUTPUT_DIR" -name '*.tmp' | wc -l | tr -d ' ')
[[ "$tmps" -eq 0 ]] && pass "no .tmp checkpoint files left" || fail "$tmps .tmp files left in $OUTPUT_DIR"

# =============================================================================
echo ""
echo "Outputs: $OUTPUT_DIR"
if [[ $FAILURES -eq 0 ]]; then echo "SMOKE TEST PASSED"; else echo "SMOKE TEST FAILED ($FAILURES checks)"; exit 1; fi
