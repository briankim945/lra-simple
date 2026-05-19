#!/bin/bash
#
# Unified Linear Probing Launch Script
# Runs grid search across multiple GPUs for any supported task
#
# Usage:
#   ./launch_linear_probe.sh --task pathfinder --data_dir /path/to/data --gpus 4
#   ./launch_linear_probe.sh --task cabc --data_dir /path/to/cabc --gpus 8 --train_split easy
#   ./launch_linear_probe.sh --task planko --data_dir /path/to/planko --models_list models.txt
#

set -e

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

TASK=""
DATA_DIR=""
TRAIN_SPLIT=""
EPOCHS=30
NUM_WORKERS=4
EXTRACT_BATCH_SIZE=64
DROPOUT_RATE=0.0
OUTPUT_DIR="results"
GPU_COUNT=4
MODELS_CSV=""
MODELS_LIST=""
SKIP_NON_224=""
SKIP_ERRORS=""
VERBOSE=""

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

print_usage() {
    echo "Usage: $0 --task TASK --data_dir PATH [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --task              Task name: pathfinder, cabc, or planko"
    echo "  --data_dir          Path to task data"
    echo ""
    echo "Optional:"
    echo "  --train_split       Training split (default: task-specific)"
    echo "  --epochs            Training epochs per config (default: 30)"
    echo "  --gpus              Number of GPUs to use (default: 4)"
    echo "  --output_dir        Output directory (default: results)"
    echo "  --models_csv        CSV file with model names (default: assets/timm_models.csv)"
    echo "  --models_list       Text file with model names (overrides --models_csv)"
    echo "  --num_workers       DataLoader workers per GPU (default: 4)"
    echo "  --extract_batch_size Batch size for feature extraction (default: 64)"
    echo "  --dropout_rate      Dropout rate for linear probe (default: 0.0)"
    echo "  --skip_non_224      Skip models with non-224x224 input size"
    echo "  --verbose           Enable detailed logging (slower, for debugging)"
    echo "  --skip_errors       Skip models logged as errors previously"
    echo ""
    echo "Examples:"
    echo "  $0 --task pathfinder --data_dir /data/pathfinder --gpus 8"
    echo "  $0 --task cabc --data_dir /data/cabc --train_split easy --gpus 4"
    echo "  $0 --task planko --data_dir /data/planko --models_list top100.txt"
    echo "  $0 --task pathfinder --data_dir /data/pf --gpus 4 --skip_non_224"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --train_split)
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --gpus)
            GPU_COUNT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --models_csv)
            MODELS_CSV="$2"
            shift 2
            ;;
        --models_list)
            MODELS_LIST="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --extract_batch_size)
            EXTRACT_BATCH_SIZE="$2"
            shift 2
            ;;
        --dropout_rate)
            DROPOUT_RATE="$2"
            shift 2
            ;;
        --skip_non_224)
            SKIP_NON_224="--skip_non_224"
            shift 1
            ;;
        --skip_errors)
            SKIP_ERRORS="--skip_errors"
            shift 1
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TASK" ]]; then
    echo "Error: --task is required"
    print_usage
    exit 1
fi

if [[ -z "$DATA_DIR" ]]; then
    echo "Error: --data_dir is required"
    print_usage
    exit 1
fi

if [[ ! "$TASK" =~ ^(pathfinder|cabc|planko|imagenet|psvrt)$ ]]; then
    echo "Error: Invalid task '$TASK'. Must be one of: pathfinder, cabc, planko, imagenet, psvrt"
    exit 1
fi

# =============================================================================
# SETUP
# =============================================================================

# Create directories
mkdir -p "${OUTPUT_DIR}/logs"

# Determine model source and count
if [[ -n "$MODELS_LIST" ]]; then
    MODEL_SOURCE="$MODELS_LIST"
    TOTAL_MODELS=$(wc -l < "$MODELS_LIST")
    MODEL_SOURCE_TYPE="list"
else
    MODEL_SOURCE="$MODELS_CSV"
    TOTAL_MODELS=$(tail -n +2 "$MODELS_CSV" | wc -l)  # Exclude header
    MODEL_SOURCE_TYPE="csv"
fi

CHUNK_SIZE=$(( (TOTAL_MODELS + GPU_COUNT - 1) / GPU_COUNT ))

# Build train_split argument
TRAIN_SPLIT_ARG=""
if [[ -n "$TRAIN_SPLIT" ]]; then
    TRAIN_SPLIT_ARG="--train_split $TRAIN_SPLIT"
fi

# Log files
LOG_FILE="${OUTPUT_DIR}/${TASK}_linear_probe.json"
CSV_FILE="${OUTPUT_DIR}/${TASK}_linear_probe.csv"

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "============================================================"
echo "Unified Linear Probing - Multi-GPU Launch"
echo "============================================================"
echo "Task:               $TASK"
echo "Data dir:           $DATA_DIR"
echo "Train split:        ${TRAIN_SPLIT:-'(task default)'}"
echo "Epochs:             $EPOCHS"
echo "Output dir:         $OUTPUT_DIR"
echo "JSON log:           $LOG_FILE"
echo "CSV log:            $CSV_FILE"
echo "Model source:       $MODEL_SOURCE ($MODEL_SOURCE_TYPE)"
echo "Total models:       $TOTAL_MODELS"
echo "GPU count:          $GPU_COUNT"
echo "Models per GPU:     ~$CHUNK_SIZE"
echo "Num workers:        $NUM_WORKERS"
echo "Extract batch size: $EXTRACT_BATCH_SIZE"
echo "Dropout rate:       $DROPOUT_RATE"
echo "============================================================"
echo ""

# =============================================================================
# GPU LAUNCH FUNCTION
# =============================================================================

run_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local gpu_log="${OUTPUT_DIR}/logs/${TASK}_lp_gpu${gpu_id}.log"
    
    echo "Launching GPU $gpu_id: models $start_idx to $((end_idx - 1))"
    
    # Build model source argument
    if [[ "$MODEL_SOURCE_TYPE" == "list" ]]; then
        MODEL_ARG="--models_list $MODEL_SOURCE"
    else
        MODEL_ARG="--models_csv $MODEL_SOURCE"
    fi
    
    python3 -m lra_simple.unified_linear_probe \
        --task "$TASK" \
        --data_dir "$DATA_DIR" \
        $TRAIN_SPLIT_ARG \
        --epochs "$EPOCHS" \
        --gpu "$gpu_id" \
        --output_dir "$OUTPUT_DIR" \
        --log_file "$LOG_FILE" \
        --csv_file "$CSV_FILE" \
        --num_workers "$NUM_WORKERS" \
        --extract_batch_size "$EXTRACT_BATCH_SIZE" \
        --dropout_rate "$DROPOUT_RATE" \
        $SKIP_NON_224 \
        $SKIP_ERRORS \
        $VERBOSE \
        $MODEL_ARG \
        --start_idx "$start_idx" \
        --end_idx "$end_idx" \
        > "$gpu_log" 2>&1
}

# =============================================================================
# LAUNCH ALL GPUS
# =============================================================================

echo "Launching $GPU_COUNT GPU processes..."
echo ""

for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    start_idx=$((gpu * CHUNK_SIZE))
    end_idx=$(( (gpu + 1) * CHUNK_SIZE ))
    
    # Cap end_idx at total models
    if [[ $end_idx -gt $TOTAL_MODELS ]]; then
        end_idx=$TOTAL_MODELS
    fi
    
    # Skip if no models for this GPU
    if [[ $start_idx -ge $TOTAL_MODELS ]]; then
        echo "GPU $gpu: No models to process"
        continue
    fi
    
    run_gpu "$gpu" "$start_idx" "$end_idx" &
done

# =============================================================================
# MONITORING INFO
# =============================================================================

echo ""
echo "All GPU processes launched!"
echo ""
echo "Monitor progress:"
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    echo "  tail -f ${OUTPUT_DIR}/logs/${TASK}_lp_gpu${gpu}.log"
done
echo ""
echo "Check JSON results:"
echo "  cat $LOG_FILE | python3 -m json.tool | head -100"
echo ""
echo "Check CSV results:"
echo "  head -20 $CSV_FILE"
echo ""
echo "Count completed:"
echo "  wc -l $CSV_FILE"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background jobs
wait

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "All GPU processes completed!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  JSON: $LOG_FILE"
echo "  CSV:  $CSV_FILE"

# Count completed models
if [[ -f "$CSV_FILE" ]]; then
    completed=$(($(wc -l < "$CSV_FILE") - 1))  # Subtract header
    echo ""
    echo "Total models completed: $completed / $TOTAL_MODELS"
fi

# Show top 10 by validation accuracy
if [[ -f "$CSV_FILE" ]]; then
    echo ""
    echo "Top 10 models by validation accuracy:"
    echo "--------------------------------------"
    head -1 "$CSV_FILE"
    tail -n +2 "$CSV_FILE" | sort -t',' -k4 -rn | head -10
fi

echo ""
echo "Done!"