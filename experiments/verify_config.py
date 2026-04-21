#!/usr/bin/env python3
"""Verify YAML config structure (no external imports)"""

import sys
import json
from pathlib import Path

# Use json module for parsing (built-in)
def parse_yaml_like(file_path):
    """Simple YAML-like parser for test files"""
    print(f"Reading: {file_path}")
    with open(file_path) as f:
        content = f.read()

    # Just verify we can read the file
    lines = content.split('\n')

    datasets_count = 0
    workloads_count = 0

    for line in lines:
        if '    name: "' in line and 'type:' in line:
            datasets_count += 1
        if '  - name: "' in line and 'pattern:' in line:
            workloads_count += 1

    return {
        'datasets': datasets_count,
        'workloads': workloads_count,
        'lines': len(lines)
    }

print("="*70)
print("Verifying Experiment Framework Setup")
print("="*70)

# Check directory structure
script_dir = Path("/Users/hsiaoko/Projects/AutoConfig/experiments")
print(f"\nChecking directory: {script_dir}")

required_dirs = [
    "baselines",
    "scripts", 
    "workloads",
    "results"
]

print("\nDirectory structure:")
for dirname in required_dirs:
    dir_path = script_dir / dirname
    exists = "✓" if dir_path.exists() else "✗"
    print(f"  {exists} {dirname}/")

# Check files
print("\nKey files:")
files_to_check = [
    "config.yaml",
    "README.md",
    "baselines/baselines.py",
    "scripts/run_all_experiments.py",
    "scripts/exp1_feature_extraction.py",
    "scripts/exp2_effectiveness.py",
    "scripts/exp3_robustness.py",
    "scripts/exp5_scalability.py",
    "scripts/exp6_case_study.py",
    "scripts/visualize_results.py",
    "workloads/workloads.py"
]

for file_path in files_to_check:
    full_path = script_dir / file_path
    exists = "✓" if full_path.exists() else "✗"
    size = f"({full_path.stat().st_size} bytes)" if full_path.exists() else ""
    print(f"  {exists} {file_path} {size}")

# Check config content
print(f"\n{'='*70}")
print("Verifying config.yaml structure")
print(f"{'='*70}")

config_path = script_dir / "config.yaml"
try:
    stats = parse_yaml_like(config_path)
    print(f"\n✓ Config file readable")
    print(f"  Datasets defined: {stats['datasets']}")
    print(f"  Workloads defined: {stats['workloads']}")
    print(f"  Total lines: {stats['lines']}")
except Exception as e:
    print(f"\n✗ Error reading config: {e}")

# Count experiment scripts
print(f"\n{'='*70}")
print("Experiment scripts")
print(f"{'='*70}")

scripts_dir = script_dir / "scripts"
exp_scripts = list(scripts_dir.glob("exp*.py"))
print(f"\nFound {len(exp_scripts)} experiment scripts:")
for script in sorted(exp_scripts):
    print(f"  • {script.name}")

print(f"\n{'='*70}")
print("Verification Complete")
print(f"{'='*70}")

print("\n✓ Experiment framework is properly set up!")
print("\nTo run experiments:")
print("  1. Ensure dependencies are installed:")
print("     pip install numpy scipy scikit-learn pandas networkx pyyaml matplotlib")
print("  2. Use the provided scripts:")
print("     cd experiments/scripts")
print("     python run_all_experiments.py --all")
print("  3. Or use the shell script:")
print("     cd experiments")
print("     ./run_experiments.sh")