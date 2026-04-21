#!/usr/bin/env python3
"""Check dependencies"""

print("Checking Python dependencies...")

packages = [
    "yaml",
    "numpy", 
    "scipy",
    "scikit learn",
    "pandas",
    "networkx",
    "matplotlib"
]

missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg} (not found)")
        missing.append(pkg)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
else:
    print("\n✓ All packages installed")