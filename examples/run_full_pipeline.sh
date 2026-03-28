#!/bin/bash
# Complete working example for AutoConfig
# This script demonstrates the full pipeline:
# 1. Extract query features
# 2. Extract graph features
# 3. Generate capacity-constrained configurations
# 4. Merge all features

set -e

echo "============================================================"
echo "AutoConfig Complete Working Example"
echo "============================================================"

# Create output directory
OUTPUT_DIR="out/example"
mkdir -p $OUTPUT_DIR

# Step 1: Extract query features
echo ""
echo "[Step 1/4] Extracting query features..."
autoconfig query \
    --input data/queries/gar_match.cu \
    --output $OUTPUT_DIR/query_features.yaml

# Step 2: Extract graph features
echo ""
echo "[Step 2/4] Extracting graph features..."
autoconfig graph \
    --input data/graph_medium_pl.csv \
    --output $OUTPUT_DIR/graph_features.yaml

# Step 3: Generate configurations (capacity-constrained)
echo ""
echo "[Step 3/4] Generating capacity-constrained configurations..."
autoconfig config \
    --capacity data/conf/system_capacity_small.yaml \
    --num-samples 5 \
    --output $OUTPUT_DIR/config_features.yaml

# Step 4: Merge all features
echo ""
echo "[Step 4/4] Merging all features..."
autoconfig merge \
    --query $OUTPUT_DIR/query_features.yaml \
    --graph $OUTPUT_DIR/graph_features.yaml \
    --config $OUTPUT_DIR/config_features.yaml \
    --output $OUTPUT_DIR/merged_features.yaml

echo ""
echo "============================================================"
echo "Example complete! Output files:"
echo "============================================================"
ls -lh $OUTPUT_DIR/

echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/query_features.yaml"
echo "  cat $OUTPUT_DIR/graph_features.yaml"
echo "  cat $OUTPUT_DIR/config_features.yaml"
echo "  cat $OUTPUT_DIR/merged_features.yaml"
