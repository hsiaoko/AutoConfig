# AConfig Experiment Framework

Complete experiment framework for evaluating AConfig along six dimensions as described in the paper.

## Overview

This experiment framework implements all experiments from the AConfig paper:

1. **Exp-1: Feature Extraction Effectiveness** - Evaluates prediction quality of the ML model
2. **Exp-2: Effectiveness** - Measures end-to-end runtime under recommended configurations
3. **Exp-3: Robustness** - Examines stability under prediction errors and distribution shift
4. **Exp-4: Efficiency** - Evaluates selection latency and tuning overhead
5. **Exp-5: Scalability and Ablation** - Studies scaling behavior and component impact
6. **Exp-6: Case Study** - Demonstrates effectiveness on graph association analytics

## Directory Structure

```
experiments/
├── config.yaml              # Main experiment configuration
├── baselines/
│   ├── __init__.py
│   └── baselines.py         # Implementation of baseline methods
├── scripts/
│   ├── run_all_experiments.py  # Main entry point
│   ├── run_single_experiment.py
│   ├── exp1_feature_extraction.py
│   ├── exp2_effectiveness.py
│   ├── exp3_robustness.py
│   ├── exp5_scalability.py
│   ├── exp6_case_study.py
│   └── visualize_results.py
├── workloads/
│   ├── __init__.py
│   └── workloads.py         # Query code templates
└── results/                 # Output directory (created automatically)
```

## Installation

### Prerequisites

```bash
# Install core dependencies
cd /path/to/AutoConfig
pip install -r requirements.txt

# Install experiment-specific dependencies
pip install matplotlib seaborn
```

### Verify Installation

```bash
cd experiments
python -c "from baselines import BayesianOptimization; print('✓ Baselines imported')"
python -c "from workloads import query_wcc; print('✓ Workloads imported')"
```

## Quick Start

### Run All Experiments

```bash
cd experiments/scripts

# Run all experiments (Exp-1 through Exp-6)
python run_all_experiments.py --all

# This will:
# 1. Train models
# 2. Workloads on all datasets
# 3. Generate JSON results in experiments/results/
# 4. Print summary statistics
```

### Run Specific Experiments

```bash
# Run specific experiments
python run_all_experiments.py --exp 1 2 3

# Or run single experiment
python run_single_experiment.py --exp 1
```

### Visualize Results

```bash
# Generate all figures
python visualize_results.py

# Figures will be saved in experiments/results/
# - fig_exp1_feature_extraction.png
# - fig_exp2_effectiveness.png
# - fig_exp3_exp4_robustness_efficiency.png
# - fig_exp5_exp6_scalability_case_study.png
```

## Experiment Details

### Exp-1: Feature Extraction Effectiveness

**Goal**: Evaluate whether the unified feature abstraction improves cost estimation quality.

**Components**:

1. **In-distribution (seen tasks)**:
   - 60/20/20 split for each workload (WCC, SSSP, PR, BFS, SubIso)
   - Compare feature variants:
     - `full`: All features (static + symbolic + graph-aware)
     - `noSPF`: Without static program features
     - `noSGF`: Without symbolic graph-parameterized features
     - `raw`: Raw tokens only (no abstraction)
   - Metrics: MAE, MAPE, R²

2. **Out-of-distribution (unseen tasks)**:
   - Transfer learning scenarios:
     - Within-class: {WCC, PR} → SSSP
     - Cross-class: {SubIso, BFS, SSSP, PR} → GARs
   - Measure MAPE degradation

**Running**:

```bash
python run_single_experiment.py --exp 1
```

**Output**:

- `exp1_inv_dist_{workload}.json` - In-distribution results
- `exp1_out_dist_{transfer}.json` - Out-of-distribution results
- `exp1_summary.json` - Complete summary

### Exp-2: Effectiveness

**Goal**: Measure end-to-end runtime achieved under recommended configurations.

**Comparison Methods**:

- **AutoConfig**: Our method
- **BO**: Bayesian Optimization
- **RL**: Reinforcement Learning
- **GPTuner**: LLM-enhanced BO
- **BestConfig**: Grid-style search
- **Oracle**: Exhaustive search (baseline, small graphs only)

**Metrics**:

- Unified cost: \(\text{cost}(C) = w_1 \cdot \text{runtime} + w_2 \cdot \text{monetary}\) with \(w_1 = w_2 = 0.5\)
- Improvement ratio over baselines
- Normalized Hamming distance to Oracle

**Running**:

```bash
python run_single_experiment.py --exp 2
```

**Output**:

- `exp2_{workload}.json` - Per-workload results
- `exp2_summary.json` - Aggregated comparison

### Exp-3: Robustness

**Goal**: Examine stability under prediction errors and distribution shift.

**Components**:

1. **Robustness to prediction errors**:
   - Simulate error rates: 0%, 10%, 20%, 30%, 50%
   - Measure cost ratio to oracle

2. **Distribution shift**:
   - Train on small graphs
   - Test on large graphs
   - Measure performance degradation

3. **Hamming distance to Oracle**:
   - How close recommendations are to optimal
   - Normalized by maximum possible distance

**Running**:

```bash
python run_single_experiment.py --exp 3
```

**Output**:

- `exp3_error_robustness.json`
- `exp3_distribution_shift.json`
- `exp3_hamming.json`

### Exp-4: Efficiency

**Goal**: Evaluate configuration selection latency and tuning overhead.

**Metrics**:

1. **Selection latency**:
   - Time to recommend a configuration (ms)
   - Measured for all methods

2. **Tuning overhead**:
   - Training/learning time
   - Resource cost during offline phase

**Running**:

```bash
python run_single_experiment.py --exp 4
```

**Output**:

- `exp4_selection_latency.json`
- `exp4_tuning_overhead.json`

### Exp-5: Scalability and Ablation

**Goal**: Study scaling behavior and impact of key components.

**Components**:

1. **Graph size scaling**:
   - Test on graphs: 1K, 5K, 10K, 50K, 100K nodes
   - Measure runtime and selection time

2. **Cluster size scaling**:
   - Test cluster sizes: 16, 32, 64, 128, 256 cores
   - Measure speedup and efficiency

3. **Component ablation**:
   - Compare variants:
     - `full`: All components
     - `noSPF`: Without static program features
     - `noSGF`: Without symbolic graph features
     - `noBk`: Without backup mechanism
   - Compute degradation percentages

**Running**:

```bash
python run_single_experiment.py --exp 5
```

**Output**:

- `exp5_graph_scaling.json`
- `exp5_cluster_scaling.json`
- `exp5_ablation.json`

### Exp-6: Case Study

**Goal**: Demonstrate effectiveness on graph association analytics (GARs).

**Application**: Fraud detection in e-commerce and financial domains.

**Components**:

1. **Configuration landscape analysis**:
   - Sample 100 configurations
   - Observe cost variation
   - Identify top configurations

2. **GARs discovery evaluation**:
   - Multi-stage pipeline:
     - Rule mining
     - Pattern matching
     - Recursive inference
   - Measure quality and runtime

3. **Multi-stage pipeline**:
   - Evaluate each stage separately
   - Total end-to-end cost

**Running**:

```bash
python run_single_experiment.py --exp 6
```

**Output**:

- `exp6_configuration_landscape.json`
- `exp6_gars_discovery.json`
- `exp6_pipeline_stages.json`

## Configuration

### Main Config File

Edit `experiments/config.yaml` to customize experiments:

```yaml
# Global settings
global:
  random_seed: 42
  num_runs: 3
  output_dir: "experiments/results"

# Datasets
datasets:
  - name: "IMDB"
    type: "knowledge_graph"
    num_vertices: 19000000
    num_edges: 100000000

  # ... more datasets

# Workloads
workloads:
  - name: "WCC"
    pattern: "iterative_fixpoint"

  # ... more workloads

# Configuration space
configuration_space:
  k_range: [1, 64]
  resources:
    - resource_type: "cpu"
      cores_range: [1, 256]

# Cost function
cost_function:
  w1: 0.5  # Runtime weight
  w2: 0.5  # Monetary cost weight
```

### Custom Configuration

```bash
# Use custom config
python run_all_experiments.py --all --config my_config.yaml

# Override output directory
python run_all_experiments.py --all --output-dir custom_results/
```

## Results

### Result Files

Results are saved as JSON files in the output directory:

```
experiments/results/
├── exp1_inv_dist_WCC.json
├── exp1_inv_dist_SSSP.json
├── exp2_effectiveness.json
├── exp3_error_robustness.json
├── exp4_selection_latency.json
├── exp5_ablation.json
├── exp6_gars_discovery.json
└── ... (more result files)
```

### Result Format

Example: `exp1_inv_dist_WCC.json`

```json
{
  "full": {
    "train": {
      "mae": 0.0123,
      "mape": 4.56,
      "r2": 0.9234
    },
    "test": {
      "mae": 0.0145,
      "mape": 5.21,
      "r2": 0.9102
    }
  },
  "noSPF": {
    "train": {...},
    "test": {...}
  }
}
```

### Visualization

Generate all figures:

```bash
cd experiments/scripts
python visualize_results.py
```

Figures (PNG, 300 DPI):

- `fig_exp1_feature_extraction.png` - Feature extraction effectiveness
- `fig_exp2_effectiveness.png` - Runtime comparison
- `fig_exp3_exp4_robustness_efficiency.png` - Robustness and efficiency
- `fig_exp5_exp6_scalability_case_study.png` - Scalability and case study

### Interpreting Results

#### MAPE (Mean Absolute Percentage Error)

Lower is better. Indicates prediction accuracy.

#### R² (R-squared)

Higher is better. Close to 1.0 means model explains most variance.

#### Unified Cost

Combines runtime and monetary cost: \(w_1 \cdot \text{runtime} + w_2 \cdot \text{monetary}\)

#### Hamming Distance

Normalized distance to Oracle configuration. Lower is better.

## Baselines

### Bayesian Optimization (BO)

Uses Gaussian Process surrogate model. Adapted to graph workloads using bag-of-words features.

### Reinforcement Learning (RL)

Trains policy offline from execution logs. Maps workload features to configuration decisions.

### GPTuner

LLM-enhanced BO. Injects domain knowledge via large language models.

### BestConfig

Grid-style search heuristic. Evaluates configurations one by one on sampled datasets.

### Oracle

Exhaustive search for provably optimal configuration. Feasible for small/medium graphs only. Serves as upper bound.

## Extending Experiments

### Adding New Workloads

Edit `experiments/workloads/workloads.py`:

```python
query_my_workload = """
__global__ void myWorkload(...) {
    // Your query code here
}
```

Then add to `experiments/workloads/__init__.py`.

### Adding Custom Metrics

Edit experiment scripts to compute additional metrics:

```python
def compute_custom_metric(results):
    # Your metric computation
    return custom_value
```

### Modifying Baselines

Implement new baseline in `experiments/baselines/baselines.py`:

```python
class MyBaseline(BaselineMethod):
    def recommend(self, workload_features):
        # Your recommendation logic
        return config
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: autoconfig`

```bash
# Make sure autoconfig is installed
pip install -e .
```

**Issue**: Out of memory on large graphs

```yaml
# In config.yaml, reduce graph sizes
datasets:
  - name: "TestGraph"
    num_vertices: 10000  # Smaller value
```

**Issue**: Experiments taking too long

```bash
# Run specific experiments only
python run_all_experiments.py --exp 1 2 3

# Reduce number of test graphs (edit in scripts)
```

### Debug Mode

```bash
# Enable verbose output
python run_single_experiment.py --exp 1 --verbose

# Check results manually
cat experiments/results/exp1_summary.json | jq .
```

## Citation

If you use this experiment framework in your research, please cite:

```bibtex
@article{autoconfig2024,
  title={AConfig: Graph Query Configuration Tuning via Hybrid Feature Abstraction},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License - See project root for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [maintainer email]

---

**Note**: The framework is designed to be self-contained and does not modify the main AutoConfig repository. All experiment code is in the `experiments/` subdirectory.