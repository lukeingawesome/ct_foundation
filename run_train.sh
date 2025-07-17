#!/usr/bin/env bash
set -euo pipefail

# Parse command line arguments
DESIRED_GPUS_ARG="4"
HELP_FLAG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            DESIRED_GPUS_ARG="$2"
            shift 2
            ;;
        -h|--help)
            HELP_FLAG=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            HELP_FLAG=true
            shift
            ;;
    esac
done

# Show help if requested or invalid arguments
if [ "$HELP_FLAG" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -g, --gpus NUMBER     Number of GPUs to use (required)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MAX_MEMORY_MB        Max memory usage for 'free' GPU (default: 2000MB)"
    echo "  MAX_UTILIZATION      Max utilization for 'free' GPU (default: 10%)"
    echo "  MIN_MEMORY_GB        Minimum GPU memory required (default: 8GB)"
    echo "  DATA_CSV             Path to data CSV file"
    echo "  PRETRAIN_CKPT        Path to pretrained checkpoint"
    echo "  OUTPUT_DIR           Output directory"
    echo ""
    echo "Examples:"
    echo "  $0 --gpus 2          # Use 2 GPUs"
    echo "  $0 -g 4              # Use 4 GPUs"
    echo "  $0 -g 1              # Single GPU training"
    exit 0
fi

# Check if GPU count was provided
if [ -z "$DESIRED_GPUS_ARG" ]; then
    echo "❌ Error: GPU count is required!"
    echo ""
    echo "Usage: $0 --gpus NUMBER"
    echo "Example: $0 --gpus 2"
    echo ""
    echo "Use $0 --help for more information"
    exit 1
fi

# Validate GPU count is a positive integer
if ! [[ "$DESIRED_GPUS_ARG" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ Error: GPU count must be a positive integer, got: '$DESIRED_GPUS_ARG'"
    exit 1
fi

# Stable distributed training script with automatic GPU detection
echo "Starting stable distributed SigLIP-CT training with automatic GPU selection..."
echo "Requested GPUs: $DESIRED_GPUS_ARG"

# Enhanced NCCL configuration for stability
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800                  # 30 minutes timeout
export NCCL_IB_TIMEOUT=1800               # InfiniBand timeout
export NCCL_SOCKET_TIMEOUT=1800           # Socket timeout
export NCCL_BLOCKING_WAIT=1               # Use blocking wait
export NCCL_ASYNC_ERROR_HANDLING=1        # Enable async error handling
export NCCL_P2P_DISABLE=0                 # Enable P2P communication
export NCCL_BUFFSIZE=8388608              # 8MB buffer size
export NCCL_NTHREADS=8                    # Number of NCCL threads

# Additional stability settings
export CUDA_LAUNCH_BLOCKING=0             # Non-blocking CUDA operations
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # Enable detailed distributed debugging
export TORCH_SHOW_CPP_STACKTRACES=1       # Show C++ stack traces on errors

# Memory and performance optimizations
export OMP_NUM_THREADS=8                  # Reduce CPU thread contention
export MKL_NUM_THREADS=8
export NCCL_RINGS=4                       # Number of NCCL rings

# Fallback settings for problematic environments
export NCCL_P2P_NET_CHUNKSIZE=2097152    # 2MB chunk size for P2P
export NCCL_SHM_DISABLE=0                 # Enable shared memory
export NCCL_NET_GDR_LEVEL=0               # Disable GPU Direct RDMA if problematic

# Configuration
DATA_CSV="${DATA_CSV:-/data/all_ct_with_labels.csv}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/model/3c_siglip/pytorch_model.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/stable_$(date +%y%m%d_%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-siglip-ct}"
WANDB_NAME="${WANDB_NAME:-stable_$(hostname -s)_$(date +%y%m%d%H%M)}"

# GPU Configuration - use command line argument
DESIRED_GPUS="$DESIRED_GPUS_ARG"           # Number of GPUs desired (from command line)
MAX_MEMORY_MB="${MAX_MEMORY_MB:-2000}"     # Max memory usage to consider GPU as "free" (MB)
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"   # Max GPU utilization to consider GPU as "free" (%)
MIN_MEMORY_GB="${MIN_MEMORY_GB:-8}"        # Minimum GPU memory required (GB)

# Conservative training parameters for stability
BATCH_SIZE=8        # Reduced from 8 to avoid OOM
EPOCHS=30
LR=2e-4
LR_MULT=0.1

# Function to get GPU information
get_gpu_info() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found. CUDA/NVIDIA drivers may not be installed."
        exit 1
    fi
    
    # Get total number of GPUs
    local total_gpus=$(nvidia-smi --list-gpus | wc -l)
    echo "Total GPUs available: $total_gpus"
    
    if [ "$total_gpus" -eq 0 ]; then
        echo "Error: No GPUs detected"
        exit 1
    fi
    
    echo "$total_gpus"
}

# Function to check if a GPU is available (low memory usage and low utilization)
is_gpu_available() {
    local gpu_id=$1
    
    # Get memory usage in MB
    local memory_used=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
    # Get memory total in MB
    local memory_total=$(nvidia-smi --id=$gpu_id --query-gpu=memory.total --format=csv,noheader,nounits)
    # Get GPU utilization percentage
    local utilization=$(nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    # Get GPU name
    local gpu_name=$(nvidia-smi --id=$gpu_id --query-gpu=name --format=csv,noheader)
    
    # Convert memory total to GB for minimum check
    local memory_total_gb=$((memory_total / 1024))
    
    echo "GPU $gpu_id ($gpu_name): ${memory_used}MB used / ${memory_total}MB total (${utilization}% util)"
    
    # Check minimum memory requirement
    if [ "$memory_total_gb" -lt "$MIN_MEMORY_GB" ]; then
        echo "  ❌ Insufficient GPU memory (${memory_total_gb}GB < ${MIN_MEMORY_GB}GB required)"
        return 1
    fi
    
    # Check if GPU is available based on memory usage and utilization
    if [ "$memory_used" -lt "$MAX_MEMORY_MB" ] && [ "$utilization" -lt "$MAX_UTILIZATION" ]; then
        echo "  ✅ GPU is available"
        return 0
    else
        echo "  ❌ GPU is busy (memory: ${memory_used}MB, util: ${utilization}%)"
        return 1
    fi
}

# Function to find available GPUs
find_available_gpus() {
    local total_gpus=$(get_gpu_info)
    local available_gpus=()
    
    echo ""
    echo "Checking GPU availability..."
    echo "Criteria: Memory < ${MAX_MEMORY_MB}MB, Utilization < ${MAX_UTILIZATION}%, Min Memory > ${MIN_MEMORY_GB}GB"
    echo ""
    
    for ((i=0; i<total_gpus; i++)); do
        if is_gpu_available $i; then
            available_gpus+=($i)
        fi
    done
    
    echo ""
    echo "Available GPUs: ${#available_gpus[@]} (${available_gpus[*]})"
    echo "Desired GPUs: $DESIRED_GPUS"
    
    if [ "${#available_gpus[@]}" -lt "$DESIRED_GPUS" ]; then
        echo ""
        echo "❌ ERROR: Insufficient available GPUs!"
        echo "   Available: ${#available_gpus[@]} GPUs (${available_gpus[*]})"
        echo "   Required:  $DESIRED_GPUS GPUs"
        echo ""
        echo "Available options:"
        echo "1. Wait for GPUs to become available"
        echo "2. Reduce DESIRED_GPUS (export DESIRED_GPUS=<number>)"
        echo "3. Adjust availability criteria:"
        echo "   - Increase MAX_MEMORY_MB (current: ${MAX_MEMORY_MB}MB)"
        echo "   - Increase MAX_UTILIZATION (current: ${MAX_UTILIZATION}%)"
        echo "   - Decrease MIN_MEMORY_GB (current: ${MIN_MEMORY_GB}GB)"
        echo ""
        echo "Current GPU status:"
        nvidia-smi
        exit 1
    fi
    
    # Select the first N available GPUs
    local selected_gpus=(${available_gpus[@]:0:$DESIRED_GPUS})
    echo "Selected GPUs: ${selected_gpus[*]}"
    
    # Set CUDA_VISIBLE_DEVICES to only use selected GPUs
    local cuda_devices=$(IFS=,; echo "${selected_gpus[*]}")
    export CUDA_VISIBLE_DEVICES="$cuda_devices"
    
    echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo ""
    
    echo "${#selected_gpus[@]}"
}

# Function to wait for GPUs to become available
wait_for_gpus() {
    local max_wait_time=1800  # 30 minutes max wait
    local wait_time=0
    local check_interval=60   # Check every minute
    
    echo "Waiting for $DESIRED_GPUS GPUs to become available..."
    echo "Will check every ${check_interval}s for up to ${max_wait_time}s..."
    
    while [ $wait_time -lt $max_wait_time ]; do
        local available_count=$(find_available_gpus 2>/dev/null || echo "0")
        
        if [ "$available_count" -ge "$DESIRED_GPUS" ]; then
            echo "✅ Required GPUs now available!"
            return 0
        fi
        
        echo "⏳ Only $available_count GPUs available, need $DESIRED_GPUS. Waiting ${check_interval}s... (${wait_time}/${max_wait_time}s elapsed)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    echo "❌ Timeout: Could not find $DESIRED_GPUS available GPUs within ${max_wait_time}s"
    return 1
}

echo ""
echo "=== TRAINING CONFIGURATION ==="
echo "  Data CSV: $DATA_CSV"
echo "  Pretrained: $PRETRAIN_CKPT"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE (per GPU)"
echo "  Requested GPUs: $DESIRED_GPUS"
echo "  Epochs: $EPOCHS"
echo "  GPU Selection Criteria:"
echo "    - Max Memory Usage: ${MAX_MEMORY_MB}MB"
echo "    - Max Utilization: ${MAX_UTILIZATION}%"
echo "    - Min GPU Memory: ${MIN_MEMORY_GB}GB"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    # Kill any remaining torchrun processes
    pkill -f "torchrun.*finetune_siglip_ct.py" || true
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}
trap cleanup EXIT

# Find and configure available GPUs
echo "=== GPU DETECTION AND SELECTION ==="
if ! NUM_GPUS=$(find_available_gpus); then
    echo ""
    echo "No sufficient GPUs available immediately. Options:"
    echo "1. Wait for GPUs (press 'w' + Enter)"
    echo "2. Exit and try later (press 'q' + Enter)"
    echo "3. Continue anyway with fewer GPUs (press 'c' + Enter)"
    
    read -p "Choice (w/q/c): " choice
    case $choice in
        w|W)
            echo "Waiting for GPUs..."
            if ! wait_for_gpus; then
                exit 1
            fi
            NUM_GPUS=$(find_available_gpus)
            ;;
        q|Q)
            echo "Exiting..."
            exit 1
            ;;
        c|C)
            echo "Attempting to continue with available GPUs..."
            # Try to find at least 1 GPU
            DESIRED_GPUS=1
            if ! NUM_GPUS=$(find_available_gpus); then
                echo "❌ No GPUs available at all!"
                exit 1
            fi
            ;;
        *)
            echo "Invalid choice. Exiting..."
            exit 1
            ;;
    esac
fi

echo ""
echo "=== FINAL CONFIGURATION ==="
echo "Using $NUM_GPUS GPUs with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo ""

# Show final GPU status for selected devices
echo "Selected GPU status:"
nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "Starting distributed training with torchrun..."

# Run with timeout and error handling
if timeout 14400 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29500 \
    --rdzv_id=stable_training \
    finetune_siglip_ct.py \
        --csv "$DATA_CSV" \
        --pretrained "$PRETRAIN_CKPT" \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --lr-backbone-mult $LR_MULT \
        --three-channel \
        --balance-sampler \
        --amp \
        --use-ema \
        --use-swa \
        --output "$OUTPUT_DIR" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-name "$WANDB_NAME" \
        --patience 10 \
        --log-level INFO \
        --debug; then
    echo "✅ Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Used GPUs: $CUDA_VISIBLE_DEVICES"
else
    exit_code=$?
    echo "❌ Training failed with exit code: $exit_code"
    
    # Provide diagnostic information
    echo ""
    echo "=== DIAGNOSTIC INFORMATION ==="
    echo "NCCL Debug logs should be above."
    echo ""
    echo "GPU Status after failure:"
    nvidia-smi --id=$CUDA_VISIBLE_DEVICES || nvidia-smi || true
    echo ""
    echo "Disk space:"
    df -h "$OUTPUT_DIR" || true
    echo ""
    echo "Check the logs in $OUTPUT_DIR for more details."
    echo ""
    echo "Common fixes:"
    echo "1. Reduce batch size (currently $BATCH_SIZE)"
    echo "2. Check if selected GPUs are still available"
    echo "3. Ensure sufficient disk space and memory"
    echo "4. Check network connectivity between nodes"
    echo "5. Try with fewer GPUs: export DESIRED_GPUS=1"
    
    exit $exit_code
fi

echo "Training script completed." 