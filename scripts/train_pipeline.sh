#!/bin/bash
# Training Pipeline Script
# Trains Bayesian models for execution time and cost prediction

set -e

echo "=============================================="
echo "AutoConfig Training Pipeline"
echo "=============================================="

# Default parameters
N_SAMPLES=${1:-500}
OUTPUT_DIR=${2:-data/models}

echo ""
echo "Parameters:"
echo "  Samples: $N_SAMPLES"
echo "  Output:  $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python -m autoconfig train --n-samples "$N_SAMPLES" --output "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Models saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review training metrics: cat $OUTPUT_DIR/training_metrics.yaml"
echo "  2. Merged pipeline: train-merged / eval-merged; recommend-conf with -o <dir> exports one YAML per picked config (see scripts/run_recommend_conf.sh)"
