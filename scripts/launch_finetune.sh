#!/bin/bash
#
# Unified Fine-tuning Launch Script
# Runs grid search across multiple GPUs for any supported task
#
# Usage:
#   ./launch_finetune.sh --task pathfinder --data_dir /path/to/data --gpus 4
#   ./launch_finetune.sh --task cabc --data_dir /path/to/cabc --gpus 8 --train_split easy
#   ./launch_finetune.sh --task planko --data_dir /path/to/planko --models_list top100.txt
#   ./launch_finetune.sh --task pathfinder --data_dir /path --gpus 4 --small_batch --gradient_checkpointing
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
OUTPUT_DIR="results"
GPU_COUNT=1
MODELS_CSV=""
MODELS_LIST=""
SKIP_NON_224=""
MAX_INPUT_SIZE=""
SMALL_BATCH=""
MICRO_BATCH_SIZE=""
GRADIENT_CHECKPOINTING=""
EARLY_STOPPING=""
PATIENCE=""
MIN_DELTA=""
VERBOSE=""

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

print_usage() {
    echo "Usage: $0 --task TASK --data_dir PATH [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --task                   Task name: pathfinder, cabc, or psvrt"
    echo "  --data_dir               Path to task data"
    echo ""
    echo "Optional:"
    echo "  --train_split            Training split (default: task-specific)"
    echo "  --epochs                 Training epochs per config (default: 30)"
    echo "  --gpus                   Number of GPUs to use (default: 4)"
    echo "  --output_dir             Output directory (default: results)"
    echo "  --models_csv             CSV file with model names (default: assets/timm_models.csv)"
    echo "  --models_list            Text file with model names (overrides --models_csv)"
    echo "  --num_workers            DataLoader workers per GPU (default: 4)"
    echo ""
    echo "Model filtering:"
    echo "  --skip_non_224           Skip models with input size > 224 (shorthand for --max_input_size 224)"
    echo "  --max_input_size N       Skip models with input size > N (overrides --skip_non_224)"
    echo ""
    echo "Memory optimization:"
    echo "  --small_batch            Use smaller batch sizes [4, 8, 16] instead of [16, 32, 64]"
    echo "  --micro_batch_size N     Actual batch size in memory (enables gradient accumulation)"
    echo "  --gradient_checkpointing Enable gradient checkpointing (saves memory, ~30% slower)"
    echo ""
    echo "Early stopping:"
    echo "  --early_stopping         Enable early stopping based on validation accuracy"
    echo "  --patience N             Epochs without improvement before stopping (default: 5)"
    echo "  --min_delta N            Minimum improvement to count as better (default: 0.1%)"
    echo ""
    echo "Debugging:"
    echo "  --verbose                Enable detailed logging (slower, for debugging)"
    echo ""
    echo "Examples:"
    echo "  $0 --task pathfinder --data_dir /data/pathfinder --gpus 8"
    echo "  $0 --task cabc --data_dir /data/cabc --train_split easy --gpus 4"
    echo "  $0 --task planko --data_dir /data/planko --models_list top100.txt"
    echo "  $0 --task pathfinder --data_dir /data/pf --gpus 4 --skip_non_224"
    echo "  $0 --task pathfinder --data_dir /data/pf --gpus 4 --small_batch --gradient_checkpointing"
    echo "  $0 --task pathfinder --data_dir /data/pf --gpus 4 --max_input_size 256"
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
        --skip_non_224)
            SKIP_NON_224="--skip_non_224"
            shift 1
            ;;
        --max_input_size)
            MAX_INPUT_SIZE="--max_input_size $2"
            shift 2
            ;;
        --small_batch)
            SMALL_BATCH="--small_batch"
            shift 1
            ;;
        --micro_batch_size)
            MICRO_BATCH_SIZE="--micro_batch_size $2"
            shift 2
            ;;
        --gradient_checkpointing)
            GRADIENT_CHECKPOINTING="--gradient_checkpointing"
            shift 1
            ;;
        --early_stopping)
            EARLY_STOPPING="--early_stopping"
            shift 1
            ;;
        --patience)
            PATIENCE="--patience $2"
            shift 2
            ;;
        --min_delta)
            MIN_DELTA="--min_delta $2"
            shift 2
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

if [[ ! "$TASK" =~ ^(pathfinder|cabc|planko)$ ]]; then
    echo "Error: Invalid task '$TASK'. Must be one of: pathfinder, cabc, planko"
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

# Determine size filter to display (max_input_size overrides skip_non_224)
if [[ -n "$MAX_INPUT_SIZE" ]]; then
    SIZE_FILTER_ARG="$MAX_INPUT_SIZE"
    SIZE_FILTER_DISPLAY="${MAX_INPUT_SIZE#--max_input_size }"
elif [[ -n "$SKIP_NON_224" ]]; then
    SIZE_FILTER_ARG="$SKIP_NON_224"
    SIZE_FILTER_DISPLAY="224 (--skip_non_224)"
else
    SIZE_FILTER_ARG=""
    SIZE_FILTER_DISPLAY="none"
fi

# Log file
LOG_FILE="${OUTPUT_DIR}/${TASK}_finetune.json"

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "============================================================"
echo "Unified Fine-tuning - Multi-GPU Launch"
echo "============================================================"
echo "Task:                  $TASK"
echo "Data dir:              $DATA_DIR"
echo "Train split:           ${TRAIN_SPLIT:-'(task default)'}"
echo "Epochs:                $EPOCHS"
echo "Output dir:            $OUTPUT_DIR"
echo "Log file:              $LOG_FILE"
echo "Model source:          $MODEL_SOURCE ($MODEL_SOURCE_TYPE)"
echo "Total models:          $TOTAL_MODELS"
echo "GPU count:             $GPU_COUNT"
echo "Models per GPU:        ~$CHUNK_SIZE"
echo "Num workers:           $NUM_WORKERS"
echo "Max input size:        $SIZE_FILTER_DISPLAY"
echo "Small batch:           ${SMALL_BATCH:-'no'}"
echo "Micro batch size:      ${MICRO_BATCH_SIZE:-'(no accumulation)'}"
echo "Gradient checkpointing: ${GRADIENT_CHECKPOINTING:-'no'}"
echo "Early stopping:        ${EARLY_STOPPING:-'no'}"
if [[ -n "$EARLY_STOPPING" ]]; then
    echo "  Patience:            ${PATIENCE:-'5 (default)'}"
    echo "  Min delta:           ${MIN_DELTA:-'0.1% (default)'}"
fi
echo "Verbose:               ${VERBOSE:-'no'}"
echo "============================================================"
echo ""

# =============================================================================
# GPU LAUNCH FUNCTION
# =============================================================================

run_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local gpu_log="${OUTPUT_DIR}/logs/${TASK}_ft_gpu${gpu_id}.log"
    
    echo "Launching GPU $gpu_id: models $start_idx to $((end_idx - 1))"
    
    # Build model source argument
    if [[ "$MODEL_SOURCE_TYPE" == "list" ]]; then
        MODEL_ARG="--models_list $MODEL_SOURCE"
    else
        MODEL_ARG="--models_csv $MODEL_SOURCE"
    fi
    
    python3 -m lra_simple.unified_finetune \
        --task "$TASK" \
        --data_dir "$DATA_DIR" \
        $TRAIN_SPLIT_ARG \
        --epochs "$EPOCHS" \
        --gpu "$gpu_id" \
        --output_dir "$OUTPUT_DIR" \
        --log_file "$LOG_FILE" \
        --num_workers "$NUM_WORKERS" \
        $SIZE_FILTER_ARG \
        $SMALL_BATCH \
        $MICRO_BATCH_SIZE \
        $GRADIENT_CHECKPOINTING \
        $EARLY_STOPPING \
        $PATIENCE \
        $MIN_DELTA \
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
    echo "  tail -f ${OUTPUT_DIR}/logs/${TASK}_ft_gpu${gpu}.log"
done
echo ""
echo "Check results:"
echo "  cat $LOG_FILE | python3 -m json.tool | head -100"
echo ""
echo "Count completed:"
echo "  python3 -c \"import json; print(len(json.load(open('$LOG_FILE'))))\""
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
echo "Results saved to: $LOG_FILE"

# Count completed models
if [[ -f "$LOG_FILE" ]]; then
    completed=$(python3 -c "import json; print(len(json.load(open('$LOG_FILE'))))" 2>/dev/null || echo "0")
    echo "Total models completed: $completed / $TOTAL_MODELS"
fi

echo ""
echo "Done!"