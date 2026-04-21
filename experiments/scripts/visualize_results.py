#!/usr/bin/env python3
"""
Visualization tool for AConfig experiment results.

Generates all figures from the paper:
- Feature extraction effectiveness (Exp-1)
- Effectiveness (Exp-2)
- Robustness and efficiency (Exp-3--4)
- Scalability and case study (Exp-5, Exp-6)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

# Paper-style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Color scheme
COLORS = {
    'full': '#1f77b4',
    'noSPF': '#ff7f0e',
    'noSGF': '#2ca02c',
    'raw': '#d62728',
    'AutoConfig': '#1f77b4',
    'BO': '#ff7f0e',
    'RL': '#2ca02c',
    'GPTuner': '#9467bd',
    'BestConfig': '#d62728',
    'Oracle': '#17becf',
}

MARKERS = {
    'full': 'o',
    'noSPF': 's',
    'noSGF': '^',
    'raw': 'v',
    'AutoConfig': 'o',
    'BO': 's',
    'RL': '^',
    'GPTuner': 'D',
    'BestConfig': 'v',
    'Oracle': '*',
}


def load_json(results_dir: Path, filename: str) -> Dict:
    """Load JSON result file."""
    path = results_dir / filename
    if not path.exists():
        print(f"Warning: {filename} not found in {results_dir}")
        return {}

    with open(path) as f:
        return json.load(f)


def create_figure_grid(n_rows: int, n_cols: int, figsize=(12, 8)) -> plt.Figure:
    """Create a figure with subplots."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


def plot_feature_extraction_effectiveness(results_dir: Path):
    """
    Figure for Exp-1: Feature extraction effectiveness.

    Creates:
    - MAPE for seen tasks
    - R² for seen tasks
    - Feature ablation (SubIso)
    - MAPE degradation: OOD tasks
    """
    print("Creating Exp-1 figures...")

    results_dir = Path(results_dir)

    # Load in-distribution results
    inv_dist_results = []
    for wl in ['WCC', 'SSSP', 'PR', 'BFS', 'SubIso']:
        data = load_json(results_dir, f"exp1_inv_dist_{wl}.json")
        inv_dist_results.append(data)

    # Load OOD results
    ood_results = load_json(results_dir, "exp1_out_dist_within_class.json")

    # Create 2x2 grid
    fig, axes = create_figure_grid(2, 2, figsize=(12, 9))

    # Workload names
    workloads = ['WCC', 'SSSP', 'PR', 'BFS', 'SubIso']
    variants = ['full', 'noSPF', 'noSGF', 'raw']
    variant_names = ['Full', 'noSPF', 'noSGF', 'Raw']

    # Plot A: MAPE for seen tasks
    ax = axes[0, 0]

    for variant in variants:
        mape_values = []
        for i, wl in enumerate(workloads):
            if inv_dist_results[i].get(variant, {}).get('test', {}):
                mape = inv_dist_results[i][variant]['test'].get('mape', 0)
                mape_values.append(mape)
            else:
                mape_values.append(0)

        ax.plot(workloads, mape_values, marker=MARKERS.get(variant, 'o'),
                color=COLORS.get(variant, 'gray'), label=variant_names[variants.index(variant)])

    ax.set_xlabel('Workload')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE: Seen Tasks')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot B: R² for seen tasks
    ax = axes[0, 1]

    for variant in variants:
        r2_values = []
        for i, wl in enumerate(workloads):
            if inv_dist_results[i].get(variant, {}).get('test', {}):
                r2 = inv_dist_results[i][variant]['test'].get('r2', 0)
                r2_values.append(r2)
            else:
                r2_values.append(0)

        ax.plot(workloads, r2_values, marker=MARKERS.get(variant, 'o'),
                color=COLORS.get(variant, 'gray'), label=variant_names[variants.index(variant)])

    ax.set_xlabel('Workload')
    ax.set_ylabel('$R^2$')
    ax.set_title('$R^2$: Seen Tasks')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot C: Feature ablation (SubIso MAPE)
    ax = axes[1, 0]

    # Use SubIso as example
    wl_idx = 4  # SubIso index
    mape_values = []
    for variant in variants:
        if inv_dist_results[wl_idx].get(variant, {}).get('test', {}):
            mape = inv_dist_results[wl_idx][variant]['test'].get('mape', 0)
            mape_values.append(mape)
        else:
            mape_values.append(0)

    bars = ax.bar(variant_names, mape_values, color=[COLORS.get(v, 'gray') for v in variants])

    ax.set_xlabel('Feature Variant')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Feature Ablation (SubIso)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot D: MAPE degradation (OOD)
    ax = axes[1, 1]

    x = [1, 2]
    x_labels = ['ID', 'OOD']
    degradation_data = [[], [], [], []]

    for variant in variants:
        if ood_results.get(variant, {}).get('id_self', {}):
            id_mape = ood_results[variant]['id_self'].get('mape', 0)
        else:
            id_mape = 0

        if ood_results.get(variant, {}).get('ood', {}):
            ood_mape = ood_results[variant]['ood'].get('mape', 0)
        else:
            ood_mape = 0

        degradation_data[variants.index(variant)] = [id_mape, ood_mape]

    width = 0.2
    for i, variant in enumerate(variants):
        ax.bar([j + i*width for j in x], degradation_data[i], width,
               label=variant_names[i], color=COLORS.get(variant, 'gray'))

    ax.set_xticks([j + width for j in x])
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Task Domain')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE Degradation: OOD')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = results_dir / "fig_exp1_feature_extraction.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    plt.close()


def plot_effectiveness(results_dir: Path):
    """
    Figure for Exp-2: Effectiveness (Runtime).

    Creates:
    - Runtime for WCC
    - Runtime for SSSP
    - Runtime for PR
    - Runtime for SubIso
    """
    print("Creating Exp-2 figures...")

    results_dir = Path(results_dir)

    # Workloads and methods
    workloads = ['WCC', 'SSSP', 'PR', 'BFS', 'SubIso']
    methods = ['AutoConfig', 'BO', 'RL', 'GPTuner', 'BestConfig']

    # Create 2x2 grid (showing 4 representative workloads)
    fig, axes = create_figure_grid(1, 4, figsize=(14, 3.5))

    display_workloads = ['WCC', 'SSSP', 'PR', 'SubIso']

    for i, wl in enumerate(display_workloads):
        ax = axes[i]

        data = load_json(results_dir, f"exp2_{wl}.json")
        if not data:
            continue

        method_avgs = []
        method_stds = []

        for method in methods:
            if method in data and data[method]:
                costs = [r['unified_cost'] for r in data[method] if r]
                method_avgs.append(np.mean(costs) if costs else 0)
                method_stds.append(np.std(costs) if costs else 0)
            else:
                method_avgs.append(0)
                method_stds.append(0)

        x_pos = np.arange(len(methods))

        bars = ax.bar(x_pos, method_avgs, yerr=method_stds,
                      color=[COLORS.get(m, 'gray') for m in methods],
                      capsize=5, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Cost (ms)')
        ax.set_title(f'{wl}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = results_dir / "fig_exp2_effectiveness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    plt.close()


def plot_robustness_and_efficiency(results_dir: Path):
    """
    Figure for Exp-3--4: Robustness and efficiency.

    Creates:
    - Hamming distance to Oracle
    - Cost ratio under shift
    - Selection latency
    - Tuning overhead
    """
    print("Creating Exp-3--4 figures...")

    results_dir = Path(results_dir)

    # Load results
    hamming_data = load_json(results_dir, "exp3_hamming.json")
    shift_data = load_json(results_dir, "exp3_distribution_shift.json")
    latency_data = load_json(results_dir, "exp4_selection_latency.json")
    overhead_data = load_json(results_dir, "exp4_tuning_overhead.json")

    # Create 2x2 grid
    fig, axes = create_figure_grid(1, 4, figsize=(14, 3.5))

    methods = ['AutoConfig', 'BO', 'RL', 'GPTuner']

    # Plot A: Hamming distance to Oracle
    ax = axes[0]

    for method in methods:
        distances = []
        for wl in ['WCC', 'SSSP', 'PR', 'BFS', 'SubIso']:
            if hamming_data.get(wl, {}).get(method, {}):
                dist = hamming_data[wl][method].get('mean_normalized', 0)
                distances.append(dist)
            else:
                distances.append(0)

        if distances:
            ax.plot(methods, distances[:len(methods)], marker=MARKERS.get(method, 'o'),
                    label=method, color=COLORS.get(method, 'gray'))

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Hamming Distance ($d_H$)')
    ax.set_title('Hamming Distance to Oracle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot B: Cost ratio under shift
    ax = axes[1]

    workloads = ['WCC', 'SSSP', 'PR', 'BFS', 'SubIso']
    shift_ratios = {method: [] for method in methods}

    for wl in workloads:
        if shift_data.get(wl):
            for method in methods:
                if method in shift_data[wl]:
                    mean_cost = shift_data[wl][method].get('mean', 0)
                    shift_ratios[method].append(mean_cost)
                else:
                    shift_ratios[method].append(0)

    for method in methods:
        if shift_ratios[method]:
            ax.plot(workloads, shift_ratios[method], marker=MARKERS.get(method, 'o'),
                    label=method, color=COLORS.get(method, 'gray'))

    ax.set_xticklabels(workloads, rotation=45, ha='right')
    ax.set_ylabel('Cost')
    ax.set_title('Cost under Distribution Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot C: Selection latency
    ax = axes[2]

    method_names = ['AutoConfig', 'BO', 'RL', 'GPTuner', 'BestConfig']

    latency_values = []
    for method in method_names:
        if latency_data.get(method, {}):
            lat = latency_data[method].get('mean_ms', 0)
            latency_values.append(lat)
        else:
            latency_values.append(0)

    bars = ax.bar(method_names, latency_values,
                  color=[COLORS.get(m, 'gray') for m in method_names])

    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.set_ylabel('Latency (ms)')
    ax.set_yscale('log')
    ax.set_title('Selection Latency')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot D: Tuning overhead
    ax = axes[3]

    overhead_values = []
    for method in method_names:
        if overhead_data.get(method, {}):
            overhead = overhead_data[method].get('mean_seconds', 0)
            overhead_values.append(overhead)
        else:
            overhead_values.append(0)

    bars = ax.bar(method_names, overhead_values,
                  color=[COLORS.get(m, 'gray') for m in method_names])

    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')
    ax.set_title('Tuning Overhead')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = results_dir / "fig_exp3_exp4_robustness_efficiency.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    plt.close()


def plot_scalability_and_case_study(results_dir: Path):
    """
    Figure for Exp-5-6: Scalability and case study.

    Creates:
    - Scalability: graph size
    - Scalability: cluster size
    - Configuration landscape: GARs
    - Runtime: GARs discovery
    """
    print("Creating Exp-5-6 figures...")

    results_dir = Path(results_dir)

    # Load results
    scaling_data = load_json(results_dir, "exp5_graph_scaling.json")
    cluster_data = load_json(results_dir, "exp5_cluster_scaling.json")
    landscape_data = load_json(results_dir, "exp6_configuration_landscape.json")
    gars_data = load_json(results_dir, "exp6_gars_discovery.json")

    # Create 2x2 grid
    fig, axes = create_figure_grid(1, 4, figsize=(14, 3.5))

    # Plot A: Scalability vs graph size
    ax = axes[0]

    graph_sizes = [1000, 5000, 10000, 50000, 100000]
    variants = ['full', 'full', 'full', 'full', 'full']

    if scaling_data:
        for wl in ['WCC', 'PR', 'BFS']:
            if wl in scaling_data:
                for variant in variants:
                    if variant in scaling_data[wl]:
                        runtimes = scaling_data[wl][variant].get('runtime', [])
                        if len(runtimes) == len(graph_sizes):
                            ax.plot(graph_sizes, runtimes, marker=MARKERS.get(variant, 'o'),
                                    label=wl, color=COLORS.get(variant, 'gray'))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Graph Size ($|V|$)')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Scalability: Graph Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot B: Scalability vs cluster size
    ax = axes[1]

    cluster_sizes = [16, 32, 64, 128, 256]

    if cluster_data:
        for wl in ['WCC', 'PR', 'BFS']:
            if wl in cluster_data:
                for variant in variants:
                    if variant in cluster_data[wl]:
                        runtimes = cluster_data[wl][variant].get('runtime', [])
                        if len(runtimes) == len(cluster_sizes):
                            ax.plot(cluster_sizes, runtimes, marker=MARKERS.get(variant, 'o'),
                                    label=wl, color=COLORS.get(variant, 'gray'))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cluster Size (CPU cores)')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Scalability: Cluster Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot C: Configuration landscape (GARs)
    ax = axes[2]

    if landscape_data and landscape_data.get('all_configs'):
        all_configs = landscape_data['all_configs']
        costs = [c['cost'] for c in all_configs]

        # Plot cost distribution
        ax.hist(costs, bins=50, alpha=0.7, color=COLORS['AutoConfig'], edgecolor='black')
        ax.set_xlabel('Cost (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Configuration Landscape: GARs')
        ax.grid(True, alpha=0.3, axis='y')

    # Plot D: GARs discovery runtime
    ax = axes[3]

    methods = ['AutoConfig', 'BO', 'RL', 'GPTuner', 'BestConfig']

    runtime_values = []
    for method in methods:
        if gars_data and method in gars_data:
            runtimes = [r['runtime'] for r in gars_data[method] if r]
            runtime_values.append(np.mean(runtimes) if runtimes else 0)
        else:
            runtime_values.append(0)

    bars = ax.bar(methods, runtime_values,
                  color=[COLORS.get(m, 'gray') for m in methods])

    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Runtime: GARs Discovery')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = results_dir / "fig_exp5_exp6_scalability_case_study.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    plt.close()


def create_all_figures(results_dir: Path, create_eps: bool = True):
    """Create all figures for the paper."""
    print("\n" + "="*70)
    print("Creating all experiment figures")
    print("="*70)

    # Check if results exist
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Create each figure
    try:
        plot_feature_extraction_effectiveness(results_dir)
    except Exception as e:
        print(f"Error creating Exp-1 figures: {e}")

    try:
        plot_effectiveness(results_dir)
    except Exception as e:
        print(f"Error creating Exp-2 figures: {e}")

    try:
        plot_robustness_and_efficiency(results_dir)
    except Exception as e:
        print(f"Error creating Exp-3-4 figures: {e}")

    try:
        plot_scalability_and_case_study(results_dir)
    except Exception as e:
        print(f"Error creating Exp-5-6 figures: {e}")

    print("\n" + "="*70)
    print("Figure creation complete")
    print("="*70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize AConfig experiment results'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Path to results directory (default: experiments/results)'
    )

    parser.add_argument(
        '--exp',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Create figure for specific experiment only'
    )

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / "results"

    # Create figures
    if args.exp:
        print(f"Creating figure for Exp-{args.exp} only...")
        # Specific experiment figure creation would go here
        create_all_figures(results_dir, create_eps=False)
    else:
        create_all_figures(results_dir)