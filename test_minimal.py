#!/usr/bin/env python3
"""Minimal test - imports only"""

import sys
from pathlib import Path

project_root = Path("/Users/hsiaoko/Projects/AutoConfig")
sys.path.insert(0, str(project_root))

print("Testing minimal imports...")

# Only test YAML and config
import yaml
print("✓ yaml")

config_path = project_root / "experiments" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"✓ Config loaded")
print(f"  Output dir: {config['global']['output_dir']}")

print("\n✓ Minimal test passed!")