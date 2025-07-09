# scripts/launch_sweep.sh
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
wandb sweep scripts/sweep.yaml | tee sweep_id.txt
SWEEP_ID=$(tail -n1 sweep_id.txt | awk '{print $NF}')
wandb agent $SWEEP_ID --count 30