# iRail Chest FM Docker Environment

This repository contains a Docker Compose setup for the iRail Chest FM project, designed for AI/ML workloads with GPU support and optimized for medical imaging tasks using MONAI.

## Prerequisites

### System Requirements

- **Operating System**: Linux (tested on Ubuntu 18.04+)
- **Docker**: Version 20.10+ 
- **Docker Compose**: Version 2.0+
- **NVIDIA GPU**: Required for training/inference
- **NVIDIA Docker Runtime**: For GPU acceleration
- **RAM**: Minimum 32GB recommended (container uses 64GB shared memory)

### Required Software Installation

1. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

2. **Install Docker Compose**:
   ```bash
   sudo apt-get update
   sudo apt-get install docker-compose-plugin
   ```

3. **Install NVIDIA Container Toolkit**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

4. **Verify GPU Access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

## Environment Setup

### Required Directory Structure

Ensure the following directories exist on your host system:

```
/BARO_Cluster/data/data/    # Read-only data directory
/home/data/                 # General data directory  
/home/model/                # Model storage directory
```

Create missing directories:
```bash
sudo mkdir -p /BARO_Cluster/data/data
sudo mkdir -p /home/data
sudo mkdir -p /home/model
sudo chown -R $USER:$USER /home/data /home/model
```

### Environment Variables

Create a `.env` file in the project root with your configuration:

```bash
# Required: Set your project name
PROJECT=your_project_name

# Optional: User configuration (defaults to current user)
UID=1000
GID=1000
USR=user

# Optional: Project root inside container  
PROJECT_ROOT=/opt/project
```

Example `.env` file:
```env
PROJECT=chest_fm_analysis
UID=1000
GID=1000
USR=researcher
PROJECT_ROOT=/opt/project
```

## Usage

### Quick Start

1. **Clone the repository and navigate to the project directory**:
   ```bash
   cd /path/to/your/project
   ```

2. **Set up environment variables**:
   ```bash
   # Method 1: Using .env file (recommended)
   echo "PROJECT=my_chest_fm_project" > .env
   
   # Method 2: Export environment variable
   export PROJECT=my_chest_fm_project
   ```

3. **Start the container**:
   ```bash
   # Using environment variable
   PROJECT=my_chest_fm_project docker compose up -d
   
   # Or if using .env file
   docker compose up -d
   ```

4. **Access the running container**:
   ```bash
   docker exec -it my_chest_fm_project /bin/bash
   ```

### Container Management

#### Start Container (Detached)
```bash
PROJECT=your_project_name docker compose up -d
```

#### Start Container (Interactive)
```bash
PROJECT=your_project_name docker compose up
```

#### Stop Container
```bash
PROJECT=your_project_name docker compose down
```

#### View Logs
```bash
PROJECT=your_project_name docker compose logs -f
```

#### Restart Container
```bash
PROJECT=your_project_name docker compose restart
```

#### Remove Container and Volumes
```bash
PROJECT=your_project_name docker compose down -v
```

## Configuration Details

### GPU Configuration

The container is configured for full GPU access:
- **Runtime**: NVIDIA Docker runtime
- **GPU Access**: All available GPUs
- **Capabilities**: Compute and utility operations
- **Environment**: Optimized for CUDA workloads

### Volume Mounts

| Host Path | Container Path | Purpose | Access |
|-----------|----------------|---------|---------|
| `.` (current directory) | `/opt/project` | Project code and workspace | Read/Write |
| `/BARO_Cluster/data/data` | `/data2` | Research datasets | Read-only |
| `/home/data` | `/data` | General data storage | Read/Write |
| `/home/model` | `/model` | Model storage | Read/Write |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT` | Required | Container name and project identifier |
| `UID` | 1000 | User ID inside container |
| `GID` | 1000 | Group ID inside container |
| `USR` | user | Username inside container |
| `PROJECT_ROOT` | /opt/project | Working directory inside container |
| `HF_HOME` | `/workspace/${PROJECT}/.cache/huggingface` | HuggingFace cache location |

### Memory Configuration

- **Shared Memory**: 64GB allocated for large-scale 3D medical imaging workloads
- **NCCL P2P**: Disabled for compatibility (`NCCL_P2P_DISABLE=1`)

## Development Workflow

### 1. Development Setup
```bash
# Set project name
export PROJECT=chest_fm_dev

# Start development container
PROJECT=$PROJECT docker compose up -d

# Access container for development
docker exec -it $PROJECT /bin/bash
```

### 2. Working with Jupyter Notebooks
```bash
# Inside container, start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

### 3. Model Training
```bash
# Inside container, verify GPU access
nvidia-smi

# Run training scripts
python train_model.py --config config/chest_fm.yaml
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check logs
PROJECT=your_project_name docker compose logs
```

#### 2. GPU Not Accessible
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker exec -it your_project_name nvidia-smi
```

#### 3. Permission Issues
```bash
# Fix ownership of mounted volumes
sudo chown -R $USER:$USER /home/data /home/model

# Check UID/GID in .env file matches your user
id -u  # Should match UID in .env
id -g  # Should match GID in .env
```

#### 4. Out of Memory Errors
```bash
# Check available memory
free -h

# Monitor container memory usage
docker stats your_project_name
```

#### 5. Volume Mount Issues
```bash
# Verify directories exist
ls -la /BARO_Cluster/data/data
ls -la /home/data
ls -la /home/model

# Check mount points inside container
docker exec -it your_project_name df -h
```

### Debugging Commands

```bash
# Check container status
docker ps -a

# Inspect container configuration
docker inspect your_project_name

# Access container with root privileges
docker exec -it --user root your_project_name /bin/bash

# Check environment variables
docker exec -it your_project_name env
```

## Performance Optimization

### GPU Memory Management
- Monitor GPU memory usage: `nvidia-smi`
- Adjust batch sizes based on available GPU memory
- Use gradient checkpointing for large models

### Storage Optimization
- Use `/data2` for read-only datasets (faster access)
- Store temporary files in `/tmp` (container local)
- Keep models in `/model` for persistence

### Network Configuration
- Container uses host networking for optimal performance
- Ensure firewall allows necessary ports for Jupyter/TensorBoard

## Security Considerations

- Container runs with user privileges (not root)
- Sensitive data should be mounted read-only when possible
- Consider using Docker secrets for API keys
- Regularly update base images for security patches

## Support

For issues related to:
- **Docker/Container**: Check Docker documentation and container logs
- **GPU Access**: Verify NVIDIA Docker runtime installation
- **Project Specific**: Refer to project-specific documentation

## License

[Include your license information here]