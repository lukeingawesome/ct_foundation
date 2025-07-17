# SigLIP-CT Distributed Training with Automatic GPU Detection

This repository contains an enhanced distributed training script for SigLIP on chest CT multilabel classification with intelligent GPU detection and allocation.

## Quick Command Reference

```bash
# Most common usage - 2 GPU training
./scripts/run_train.sh --gpus 2

# Single GPU training  
./scripts/run_train.sh --gpus 1

# Multi-GPU training (4 GPUs)
./scripts/run_train.sh --gpus 4

# Show all available options
./scripts/run_train.sh --help
```

## Features

- **Automatic GPU Detection**: Intelligently finds and allocates available GPUs based on memory usage and utilization
- **Flexible GPU Requirements**: Command-line specification of desired GPU count
- **Smart Availability Criteria**: Configurable thresholds for determining GPU availability
- **Interactive Fallback Options**: User choice when insufficient GPUs are available
- **Distributed Training Support**: Multi-GPU training with NCCL optimizations
- **Error Recovery**: Robust error handling and diagnostic information

## Quick Start

### Basic Usage

```bash
# Standard 2-GPU training
./scripts/run_train.sh --gpus 2

# Single GPU training
./scripts/run_train.sh --gpus 1

# High-end 4-GPU training
./scripts/run_train.sh --gpus 4
```

### Show Help

```bash
./scripts/run_train.sh --help
```

## GPU Detection Logic

The script automatically detects available GPUs based on three criteria:

1. **Memory Usage**: GPU memory usage must be below threshold (default: 10000MB)
2. **Utilization**: GPU utilization must be below threshold (default: 10%)
3. **Minimum Memory**: GPU must have sufficient total memory (default: 8GB)

### Example GPU Detection Output

```
=== GPU DETECTION AND SELECTION ===

Total GPUs available: 4

Checking GPU availability...
Criteria: Memory < 10000MB, Utilization < 10%, Min Memory > 8GB

GPU 0 (NVIDIA RTX 4090): 1234MB used / 24564MB total (5% util)
  ✅ GPU is available
GPU 1 (NVIDIA RTX 4090): 3456MB used / 24564MB total (85% util)  
  ❌ GPU is busy (memory: 3456MB, util: 85%)
GPU 2 (NVIDIA RTX 4090): 890MB used / 24564MB total (2% util)
  ✅ GPU is available
GPU 3 (NVIDIA RTX 4090): 567MB used / 24564MB total (1% util)
  ✅ GPU is available

Available GPUs: 3 (0 2 3)
Desired GPUs: 2
Selected GPUs: 0 2
Set CUDA_VISIBLE_DEVICES=0,2
```

## Configuration Options

### Command Line Arguments

| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--gpus` | `-g` | Number of GPUs to use | ✅ Yes |
| `--help` | `-h` | Show help message | No |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_MEMORY_MB` | 10000 | Max memory usage for 'free' GPU (MB) |
| `MAX_UTILIZATION` | 10 | Max utilization for 'free' GPU (%) |
| `MIN_MEMORY_GB` | 8 | Minimum GPU memory required (GB) |
| `DATA_CSV` | `/data/all_ct_with_labels.csv` | Path to data CSV file |
| `PRETRAIN_CKPT` | `/model/3c_siglip/pytorch_model.bin` | Path to pretrained checkpoint |
| `OUTPUT_DIR` | `runs/stable_YYMMDD_HHMM` | Output directory |
| `WANDB_PROJECT` | `siglip-ct` | Weights & Biases project name |
| `WANDB_NAME` | Auto-generated | Weights & Biases run name |

## Usage Examples

### 1. Standard Multi-GPU Training

```bash
./scripts/run_train.sh --gpus 2
```

### 2. Relaxed GPU Availability Criteria

Allow GPUs with higher memory usage and utilization:

```bash
export MAX_MEMORY_MB=4000     # Allow up to 4GB used
export MAX_UTILIZATION=20     # Allow up to 20% utilization
export MIN_MEMORY_GB=6        # Require at least 6GB memory

./scripts/run_train.sh --gpus 2
```

### 3. Custom Data and Model Paths

```bash
export DATA_CSV="/path/to/your/data.csv"
export PRETRAIN_CKPT="/path/to/your/model.bin"
export OUTPUT_DIR="/path/to/output"

./scripts/run_train.sh --gpus 4
```

### 4. Single GPU Training

```bash
./scripts/run_train.sh --gpus 1
```

## Interactive Options

When insufficient GPUs are available, the script provides interactive options:

```
❌ ERROR: Insufficient available GPUs!
   Available: 1 GPUs (3)
   Required:  2 GPUs

Available options:
1. Wait for GPUs to become available
2. Reduce DESIRED_GPUS (export DESIRED_GPUS=<number>)
3. Adjust availability criteria:
   - Increase MAX_MEMORY_MB (current: 2000MB)
   - Increase MAX_UTILIZATION (current: 10%)
   - Decrease MIN_MEMORY_GB (current: 8GB)

No sufficient GPUs available immediately. Options:
1. Wait for GPUs (press 'w' + Enter)
2. Exit and try later (press 'q' + Enter)  
3. Continue anyway with fewer GPUs (press 'c' + Enter)

Choice (w/q/c):
```

### Option Details

- **Wait (w)**: Script monitors GPU availability for up to 30 minutes
- **Quit (q)**: Exit and try again later
- **Continue (c)**: Use fewer GPUs than requested (minimum 1)

## Error Handling

### Common Error Messages

#### 1. No GPU Count Specified
```bash
❌ Error: GPU count is required!

Usage: ./run_stable_distributed.sh --gpus NUMBER
Example: ./run_stable_distributed.sh --gpus 2
```

**Solution**: Always specify the `--gpus` argument.

#### 2. Invalid GPU Count
```bash
❌ Error: GPU count must be a positive integer, got: 'abc'
```

**Solution**: Use a positive integer (1, 2, 3, etc.).

#### 3. No GPUs Available
```bash
❌ No GPUs available at all!
```

**Solution**: Check if nvidia-smi works and GPUs are not in use.

### GPU Status Checking

Check current GPU availability:

```bash
# Check manually with nvidia-smi
nvidia-smi

# Or test the training script with dry-run
./scripts/run_train.sh --gpus 1 --help
```

## Training Configuration

The script uses the following default training parameters:

- **Batch Size**: 8 per GPU (adjustable via script)
- **Epochs**: 30
- **Learning Rate**: 2e-4
- **Backbone LR Multiplier**: 0.1
- **AMP**: Enabled (Automatic Mixed Precision)
- **EMA**: Enabled (Exponential Moving Average)
- **SWA**: Enabled (Stochastic Weight Averaging)

## Output

### Successful Training

```
✅ Training completed successfully!
Results saved to: runs/stable_240315_1430
Used GPUs: 0,2
```

### Training Files

The output directory contains:
- `best_crg.pth`: Best model checkpoint (highest CRG score)
- `last_epoch.pth`: Final epoch checkpoint
- Training logs and metrics
- WandB logs (if configured)

## Troubleshooting

### 1. CUDA Out of Memory

- Reduce batch size in the script
- Use fewer GPUs
- Increase `MAX_MEMORY_MB` threshold

### 2. No Available GPUs

- Wait for current jobs to finish
- Increase `MAX_MEMORY_MB` and `MAX_UTILIZATION`
- Reduce `MIN_MEMORY_GB` requirement

### 3. NCCL Errors

- Check network connectivity between GPUs
- Verify CUDA and NCCL installation
- Try single GPU training first

### 4. Permission Errors

```bash
chmod +x scripts/run_train.sh
```

## Dependencies

- PyTorch with CUDA support
- NVIDIA GPU drivers
- NCCL for distributed training
- nvidia-smi command-line tool
- Python packages: numpy, pandas, sklearn, transformers, wandb, tqdm

## Advanced Usage

### Custom NCCL Configuration

The script includes optimized NCCL settings for stability:

```bash
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# ... and more
```

### Debugging

Enable detailed logging:

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
./scripts/run_train.sh --gpus 2
```

## Contributing

To modify GPU detection logic, edit the functions in `scripts/run_train.sh`:

- `get_gpu_info()`: Get total GPU count
- `is_gpu_available()`: Check individual GPU availability  
- `find_available_gpus()`: Find and select GPUs
- `wait_for_gpus()`: Wait for GPUs to become available

## License

[Add your license information here]
