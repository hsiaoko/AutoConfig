#!/usr/bin/env python3
"""
Main entry point for running all AConfig experiments.

Runs:
- Exp-1: Feature Extraction Effectiveness
- Exp-2: Effectiveness (Runtime)
- Exp-3: Robustness
- Exp-4: Efficiency
- Exp-5: Scalability and Ablation
- Exp-6: Case Study (GARs)
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_exp1(config: dict):
    """Run Exp-1: Feature Extraction Effectiveness"""
    print("\n" + "="*70)
    print("Running Exp-1: Feature Extraction Effectiveness")
    print("="*70)

    # Import as module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp1_feature_extraction",
        Path(__file__).parent / "exp1_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_exp1(config)


def run_exp2(config: dict):
    """Run Exp-2: Effectiveness"""
    print("\n" + "="*70)
    print("Running Exp-2: Effectiveness (End-to-end Runtime)")
    print("="*70)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp2_effectiveness",
        Path(__file__).parent / "exp2_effectiveness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_exp2(config)


def run_exp3_exp4(config: dict):
    """Run Exp-3: Robustness and Exp-4: Efficiency"""
    print("\n" + "="*70)
    print("Running Exp-3: Robustness and Exp-4: Efficiency")
    print("="*70)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp3_robustness",
        Path(__file__).parent / "exp3_robustness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_exp3_exp4(config)


def run_exp5(config: dict):
    """Run Exp-5: Scalability and Ablation"""
    print("\n" + "="*70)
    print("Running Exp-5: Scalability and Ablation")
    print("="*70)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp5_scalability",
        Path(__file__).parent / "exp5_scalability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_exp5(config)


def run_exp6(config: dict):
    """Run Exp-6: Case Study (GARs)"""
    print("\n" + "="*70)
    print("Running Exp-6: Case Study - Graph Association Rules")
    print("="*70)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp6_case_study",
        Path(__file__).parent / "exp6_case_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_exp6(config)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run AConfig experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_all_experiments.py --all

  # Run specific experiments
  python run_all_experiments.py --exp 1 2 3

  # Run with custom config
  python run_all_experiments.py --all --config custom_config.yaml
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments (Exp-1 through Exp-6)'
    )

    parser.add_argument(
        '--exp',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4, 5, 6],
        help='Specific experiments to run (1-6)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file (default: experiments/config.yaml)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what will be run without executing'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    if args.config:
        config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override output dir if specified
    if args.output_dir:
        config['global']['output_dir'] = args.output_dir

    # Determine which experiments to run
    experiments_to_run = []

    if args.all:
        experiments_to_run = [1, 2, 3, 4, 5, 6]
    elif args.exp:
        experiments_to_run = args.exp
    else:
        print("Error: Must specify either --all or --exp")
        parser.print_help()
        sys.exit(1)

    # Print what will be run
    print("\n" + "#"*70)
    print("# AConfig Experiment Runner")
    print("#"*70)
    print(f"\nConfig: {config_path}")
    print(f"Output directory: {config['global']['output_dir']}")
    print(f"\nExperiments to run:")
    for exp_num in experiments_to_run:
        print(f"  - Exp-{exp_num}")

    if args.dry_run:
        print("\nDry run mode - exiting without execution")
        return

    # Run experiments
    exp_functions = {
        1: run_exp1,
        2: run_exp2,
        3: lambda c: run_exp3_exp4(c),  # Combined
        4: lambda c: run_exp3_exp4(c),  # Combined
        5: run_exp5,
        6: run_exp6,
    }

    # Run in order, avoiding duplicates
    unique_exps = sorted(set(experiments_to_run))

    for exp_num in unique_exps:
        try:
            exp_functions[exp_num](config)
        except Exception as e:
            print(f"\nError running Exp-{exp_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "="*70)
    print("Experiment Run Complete")
    print("="*70)
    print(f"\nResults saved to: {config['global']['output_dir']}")
    print("\nTo visualize results, run:")
    print(f"  python {Path(__file__).parent}/visualize_results.py")


if __name__ == '__main__':
    main()