# iRail Chest FM Docker Environment

## Setup

```bash
git clone https://github.com/lukeingawesome/ct_foundation.git
cd ct_foundation
```

## Quick Start

```bash
PROJECT={my_project} docker compose up -d
```

## Usage

### Start Container
```bash
PROJECT=your_project_name docker compose up -d
```

### Access Container
```bash
docker exec -it your_project_name /bin/bash
```

### Stop Container
```bash
PROJECT=your_project_name docker compose down
```

### View Logs
```bash
PROJECT=your_project_name docker compose logs -f
```

## Environment Variables

Create a `.env` file with your configuration:

```env
PROJECT=your_project_name
UID=1000
GID=1000
USR=user
PROJECT_ROOT=/opt/project
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `.` (current directory) | `/opt/project` | Project workspace |
| `/BARO_Cluster/data/data` | `/data2` | Research datasets (read-only) |
| `/home/data` | `/data` | General data storage |
| `/home/model` | `/model` | Model storage |

## Troubleshooting

### Check Container Status
```bash
docker ps -a
```

### Check GPU Access
```bash
docker exec -it your_project_name nvidia-smi
```
