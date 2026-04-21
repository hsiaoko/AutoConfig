#!/bin/bash
# Quick start script for running AConfig experiments

set -e

echo "=========================================="
echo "AutoConfig Experiment Runner"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# Check if matplotlib is installed
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "Installing matplotlib..."
    pip3 install matplotlib seaborn
fi

# Default experiment
EXP_NUM="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)
            EXP_NUM="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --visualize-only)
            VISUALIZE_ONLY="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp <NUM>         Run specific experiment (1-6) or 'all'"
            echo "  --config <FILE>      Use custom config file"
            echo "  --output-dir <DIR>   Override output directory"
            echo "  --visualize-only     Only generate plots (requires existing results)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all experiments"
            echo "  $0 --exp 1            # Run Exp-1 only"
            echo "  $0 --exp 1 2 3        # Run Exp-1, Exp-2, Exp-3"
            echo "  $0 --visualize-only   # Generate plots only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD=("python3" "scripts/run_all_experiments.py")

if [ "$EXP_NUM" = "all" ]; then
    CMD+=("--all")
else
    CMD+=("--exp")
    IFS=' ' read -ra EXP_ARRAY <<< "$EXP_NUM"
    for exp in "${EXP_ARRAY[@]}"; do
        CMD+=("$exp")
    done
fi

if [ -n "$CONFIG_FILE" ]; then
    CMD+=("--config" "$CONFIG_FILE")
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=("--output-dir" "$OUTPUT_DIR")
fi

# Run experiments
if [ "$VISUALIZE_ONLY" = "true" ]; then
    echo ""
    echo "Running visualization only..."
    echo ""
    python3 scripts/visualize_results.py "${OUTPUT_DIR:-results}"
else
    echo ""
    echo "Running experiments: $EXP_NUM"
    echo ""

    # Run experiments
    "${CMD[@]}"

    echo ""
    echo "=========================================="
    echo "Experiments completed!"
    echo "=========================================="

    # Ask about visualization
    echo ""
    read -p "Generate plots now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Generating plots..."
        python3 scripts/visualize_results.py "${OUTPUT_DIR:-results}"
    fi
fi

echo ""
echo "Done! Check the results in: ${OUTPUT_DIR:-experiments/results}"
echo ""