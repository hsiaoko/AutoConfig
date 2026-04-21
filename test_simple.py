#!/usr/bin/env python3
"""Simple diagnostic test"""

import sys
import os
from pathlib import Path

print("Python version:", sys.version)
print("Working directory:", os.getcwd())

# Test basic imports
print("\nTesting imports...")

try:
    import yaml
    print("✓ yaml")
except Exception as e:
    print("✗ yaml:", e)

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print("✗ numpy:", e)

try:
    import networkx as nx
    print("✓ networkx")
except Exception as e:
    print("✗ networkx:", e)

try:
    import matplotlib
    print("✓ matplotlib")
except Exception as e:
    print("✗ matplotlib:", e)

# Test reading autoconfig modules
print("\nTesting autoconfig modules...")
project_root = Path("/Users/hsiaoko/Projects/AutoConfig")
sys.path.insert(0, str(project_root))

try:
    from autoconfig import CostPredictor
    print("✓ CostPredictor")
except Exception as e:
    print("✗ CostPredictor:", type(e).__name__, str(e))

try:
    from experiments.baselines import BayesianOptimization
    print("✓ BayesianOptimization")
except Exception as e:
    print("✗ BayesianOptimization:", type(e).__name__, str(e))

# Test config loading
print("\nTesting config...")
try:
    config_path = project_root / "experiments" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded with {len(config['datasets'])} datasets")
except Exception as e:
    print("✗ Config:", type(e).__name__, str(e))

print("\n✓ Basic test completed!")