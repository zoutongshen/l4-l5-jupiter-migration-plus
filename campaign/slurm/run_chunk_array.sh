#!/bin/bash
# ALICE job array: one task per chunk of a prepared run, wrapping s3_run_chunk.py.
# UNTESTED TEMPLATE: the ALICE environment is set up after the laptop pilot.
#
#   export TROJAN_DATA=/path/on/alice/trojans        # data root (raw/, initial/, runs/)
#   python campaign/scripts/s3_launch_local.py <config> --prepare-only
#   sbatch --array=0-<n_chunks-1> campaign/slurm/run_chunk_array.sh <run_id>   # from the repo root
#
# Resubmitting the same array resumes partial chunks from their last checkpoint.
# Logs go to $TROJAN_DATA/runs/<run_id>/logs/, never to the submit directory.

#SBATCH --job-name=trojans
#SBATCH --partition=cpu-long
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
RUN_ID="$1"
: "${TROJAN_DATA:?set TROJAN_DATA to the data root}"
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"

CHUNK="$SLURM_ARRAY_TASK_ID"
TAG="c$(printf %02d "$CHUNK")"
RUN_DIR="$TROJAN_DATA/runs/$RUN_ID"
RESUME=""
[ -f "$RUN_DIR/chunks/$TAG/latest.json" ] && RESUME="--resume"

# module load Miniconda3 && conda activate trojan-rb    # adjust to ALICE's module names
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python "$REPO/campaign/scripts/s3_run_chunk.py" --run-id "$RUN_ID" --chunk "$CHUNK" $RESUME \
    >> "$RUN_DIR/logs/$TAG.log" 2>&1
