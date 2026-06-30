#!/bin/bash
set -e
PYTHON=/home/golobs/miniconda3/envs/DeepDTAGen/bin/python
cd /home/golobs/DeepDTAGen

echo "=== PIPELINE START: $(date) ==="

echo ""
echo "=== STEP 1: TRAINING (davis, 5 epochs, eval every epoch) ==="
$PYTHON -u training.py davis --epochs 5 --eval-interval 1

echo ""
echo "=== STEP 2: INFERENCE - FROM-SCRATCH MODEL ==="
$PYTHON -u test.py --dataset davis --model-path saved_models/deepdtagen_model_davis.pth

echo ""
echo "=== STEP 3: INFERENCE - PRE-TRAINED MODEL (baseline comparison) ==="
$PYTHON -u test.py --dataset davis

echo ""
echo "=== PIPELINE COMPLETE: $(date) ==="
echo ""
echo "--- Saved model ---"
ls -lh saved_models/
echo "--- Affinity outputs ---"
ls -lh Affinities/
