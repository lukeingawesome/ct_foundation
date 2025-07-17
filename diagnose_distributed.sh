#!/usr/bin/env bash
set -euo pipefail

echo "=== Distributed Training Diagnostics ==="
echo "Timestamp: $(date)"
echo ""

# Check GPU availability
echo "1. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv
    echo ""
    echo "GPU Memory Details:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | \
        awk 'BEGIN{print "GPU Used(MB) Free(MB) Total(MB) Usage%"} {printf "%-3d %-8d %-8d %-9d %.1f%%\n", NR-1, $1, $2, $3, ($1/$3)*100}'
else
    echo "ERROR: nvidia-smi not found"
fi

echo ""
echo "2. System Resources:"
echo "Memory usage:"
free -h
echo ""
echo "Disk space:"
df -h /tmp /data || true
echo ""
echo "CPU load:"
uptime
echo ""

# Check network configuration
echo "3. Network Configuration:"
echo "Hostname: $(hostname)"
echo "IP addresses:"
ip addr show | grep -E "inet [0-9]" | head -5
echo ""

# Check PyTorch and CUDA
echo "4. PyTorch/CUDA Environment:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'Distributed available: {torch.distributed.is_available()}')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
" 2>/dev/null || echo "ERROR: Could not import PyTorch"

echo ""
echo "5. Process Information:"
echo "Current PyTorch processes:"
ps aux | grep -E "(python|torchrun)" | grep -v grep || echo "No PyTorch processes found"
echo ""

# Check environment variables
echo "6. Environment Variables:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "NCCL_DEBUG: ${NCCL_DEBUG:-not set}"
echo "WORLD_SIZE: ${WORLD_SIZE:-not set}"
echo "RANK: ${RANK:-not set}"
echo "LOCAL_RANK: ${LOCAL_RANK:-not set}"
echo ""

# Check for common issues
echo "7. Common Issues Check:"

# Check for zombie processes
zombie_count=$(ps aux | awk '$8 ~ /^Z/ { count++ } END { print count+0 }')
if [ "$zombie_count" -gt 0 ]; then
    echo "⚠️  WARNING: $zombie_count zombie processes detected"
else
    echo "✅ No zombie processes"
fi

# Check disk space
root_space=$(df / | awk 'NR==2 {print $(NF-1)}' | sed 's/%//')
if [ "$root_space" -gt 90 ]; then
    echo "⚠️  WARNING: Root filesystem is ${root_space}% full"
else
    echo "✅ Sufficient disk space"
fi

# Check memory
mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
if [ "$mem_usage" -gt 90 ]; then
    echo "⚠️  WARNING: Memory usage is ${mem_usage}%"
else
    echo "✅ Memory usage acceptable (${mem_usage}%)"
fi

echo ""
echo "8. Recommended Actions:"
echo "If experiencing NCCL timeouts:"
echo "  - Reduce batch size (try 4 or 6 instead of 8)"
echo "  - Increase NCCL_TIMEOUT (currently set to 30 minutes in stable script)"
echo "  - Check network connectivity between GPUs"
echo "  - Try running: ./run_stable_distributed.sh"
echo ""
echo "If experiencing OOM errors:"
echo "  - Reduce batch size"
echo "  - Disable gradient accumulation"
echo "  - Use FP16 precision (--amp flag)"
echo ""
echo "If experiencing hanging:"
echo "  - Kill all torchrun processes: pkill -f torchrun"
echo "  - Clear GPU memory: python3 -c 'import torch; torch.cuda.empty_cache()'"
echo "  - Check for hardware issues"

echo ""
echo "=== Diagnostic Complete ===" 