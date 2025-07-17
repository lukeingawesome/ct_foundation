#!/usr/bin/env bash

# Example usage of the enhanced GPU detection script

echo "=== GPU Detection Examples ==="
echo ""

# Example 1: Request 2 GPUs (most common)
echo "1. Standard multi-GPU training (2 GPUs):"
echo "   ./run_stable_distributed.sh --gpus 2"
echo ""

# Example 2: Request specific number of GPUs
echo "2. Request specific number of GPUs:"
echo "   ./run_stable_distributed.sh --gpus 4"
echo "   ./run_stable_distributed.sh -g 8"
echo ""

# Example 3: Single GPU training
echo "3. Single GPU training:"
echo "   ./run_stable_distributed.sh --gpus 1"
echo ""

# Example 4: Adjust availability criteria
echo "4. Relax GPU availability criteria:"
echo "   export MAX_MEMORY_MB=4000    # Allow GPUs with up to 4GB used"
echo "   export MAX_UTILIZATION=20    # Allow GPUs with up to 20% utilization" 
echo "   export MIN_MEMORY_GB=6       # Require at least 6GB GPU memory"
echo "   ./run_stable_distributed.sh --gpus 2"
echo ""

# Example 5: Show help
echo "5. Show help and available options:"
echo "   ./run_stable_distributed.sh --help"
echo ""

# Example 5: Check what GPUs are currently available
echo "5. Check current GPU availability:"
echo ""

# Function to show current GPU status
show_gpu_status() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not available"
        return 1
    fi
    
    echo "Current GPU Status:"
    echo "=================="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    echo ""
    
    echo "Detailed availability check (using default criteria):"
    echo "Memory < 2000MB, Utilization < 10%, Min Memory > 8GB"
    echo "---------------------------------------------------"
    
    local total_gpus=$(nvidia-smi --list-gpus | wc -l)
    local available_count=0
    
    for ((i=0; i<total_gpus; i++)); do
        local memory_used=$(nvidia-smi --id=$i --query-gpu=memory.used --format=csv,noheader,nounits)
        local memory_total=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader,nounits)
        local utilization=$(nvidia-smi --id=$i --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        local gpu_name=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader)
        local memory_total_gb=$((memory_total / 1024))
        
        printf "GPU %d (%s): %dMB used / %dMB total (%d%% util) " $i "$gpu_name" $memory_used $memory_total $utilization
        
        if [ "$memory_total_gb" -lt 8 ]; then
            echo "❌ Insufficient memory (${memory_total_gb}GB < 8GB)"
        elif [ "$memory_used" -lt 2000 ] && [ "$utilization" -lt 10 ]; then
            echo "✅ Available"
            available_count=$((available_count + 1))
        else
            echo "❌ Busy"
        fi
    done
    
    echo ""
    echo "Summary: $available_count out of $total_gpus GPUs are available"
}

# Show current status
show_gpu_status

echo ""
echo "=== Command Line Arguments ==="
echo "--gpus, -g NUMBER    - Number of GPUs to request (required)"
echo "--help, -h           - Show help message"
echo ""
echo "=== Environment Variables (Optional) ==="
echo "MAX_MEMORY_MB        - Max memory usage for 'free' GPU (default: 2000MB)"  
echo "MAX_UTILIZATION      - Max utilization for 'free' GPU (default: 10%)"
echo "MIN_MEMORY_GB        - Minimum GPU memory required (default: 8GB)"
echo "DATA_CSV             - Path to data CSV file"
echo "PRETRAIN_CKPT        - Path to pretrained checkpoint"
echo "OUTPUT_DIR           - Output directory"
echo ""
echo "=== Interactive Options ==="
echo "When insufficient GPUs are available, the script will offer:"
echo "- Wait for GPUs to become available (up to 30 minutes)"
echo "- Exit and try later"  
echo "- Continue with fewer GPUs (minimum 1)"
echo "" 