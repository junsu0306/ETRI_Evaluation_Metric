#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=== MEMORY ==='
free -h

printf '%s\n' '=== DISK ==='
df -h "${1:-.}"

printf '%s\n' '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader

printf '%s\n' '=== GPU PROCESSES ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
