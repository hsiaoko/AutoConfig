#!/usr/bin/env python3
"""Simple test script for Exp-1"""

import os
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules directly
print("Importing modules...")

# Import yaml
import yaml
print("✓ yaml imported")

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
print(f"✓ Config loaded: {config_path}")
print(f"  Output dir: {config['global']['output_dir']}")

# Create output dir
output_dir = Path(config['global']['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Output dir created: {output_dir}")

print("\n✓ All basic imports and setup successful!")
print("\nNow testing experiment script...")

# Import experiment functions
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from exp1_feature_extraction import run_exp1
print("✓ exp1_feature_extraction imported")

# Run Exp-1
print("\n" + "="*70)
print("Running Exp-1 (Simplified Test)")
print("="*70)

run_exp1(config)

print("\n" + "="*70)
print("Exp-1 completed!")
print("="*70)