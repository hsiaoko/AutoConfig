#!/usr/bin/env python3
"""
Convenience script to run a single experiment.

Usage:
  python run_single_experiment.py --exp 1
  python run_single_experiment.py --exp 6 --config my_config.yaml
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description='Run a single AConfig experiment'
    )

    parser.add_argument(
        '--exp',
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help='Experiment number to run'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    args = parser.parse_args()

    # Build command for run_all_experiments.py
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_all_experiments.py"),
        "--exp", str(args.exp)
    ]

    if args.config:
        cmd.extend(["--config", args.config])

    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])

    # Execute
    import subprocess
    result = subprocess.run(cmd)

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()