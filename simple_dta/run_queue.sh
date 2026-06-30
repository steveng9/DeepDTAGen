#!/bin/bash
# Run a batch of train.py jobs with bounded concurrency, fully detached.
#   bash run_queue.sh <jobs_file> [max_parallel] [threads_per_job]
# jobs_file: one job per line, "<logtag>|<train.py args>".  Lines starting # skipped.
# Each job logs to runs/<logtag>.log. Progress to runs/queue.driver.log.
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepDTAGen
mkdir -p runs

JOBS_FILE=${1:?usage: run_queue.sh <jobs_file> [max_parallel] [threads]}
MAXP=${2:-4}
THREADS=${3:-12}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
echo "QUEUE START $(date) | file=$JOBS_FILE max_parallel=$MAXP threads=$THREADS" \
    >> runs/queue.driver.log

while IFS='|' read -r tag args; do
    case "$tag" in ''|\#*) continue;; esac
    tag=$(echo "$tag" | xargs)   # trim whitespace
    while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
    echo "LAUNCH $(date) | $tag |$args" >> runs/queue.driver.log
    ( python -u train.py $args > "runs/${tag}.log" 2>&1
      echo "DONE   $(date) | $tag" >> runs/queue.driver.log ) &
done < "$JOBS_FILE"

wait
echo "QUEUE ALL DONE $(date)" >> runs/queue.driver.log
