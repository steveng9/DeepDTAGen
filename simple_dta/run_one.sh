#!/bin/bash
# Launch one train.py run with the conda env activated, logging to runs/<logtag>.log.
# Meant to be wrapped in setsid/nohup so it survives logout. Thread count is
# inherited from OMP_NUM_THREADS/MKL_NUM_THREADS so several can share the cores.
#   bash run_one.sh <logtag> <train.py args...>
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepDTAGen

LOGTAG=${1:?usage: run_one.sh <logtag> <train.py args...>}; shift
mkdir -p runs
echo "START $(date) | logtag=$LOGTAG | args: $* | OMP=${OMP_NUM_THREADS:-default}" \
    >> runs/run_one.driver.log
python -u train.py "$@" > "runs/${LOGTAG}.log" 2>&1
echo "DONE  $(date) | logtag=$LOGTAG" >> runs/run_one.driver.log
