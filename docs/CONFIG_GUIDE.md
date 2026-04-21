# Configuration generation

This guide explains how **configuration candidates** are produced: default catalog, CLI resource hints, custom resource YAML, and how that relates to `experiments/scripts/step3_system_config.py`.

## Background

A bare command such as:

```bash
autoconfig config -n 10 -o out/config.yaml
```

uses the **built-in catalog** wired into `cmd_config` in `autoconfig/cli.py`. For **explicit** catalog selection or **CPU/memory/GPU**-based simple catalogs, use the pipeline script:

```bash
python experiments/scripts/step3_system_config.py --help
```

## Option 1: Default catalog (quick test)

```bash
python experiments/scripts/step3_system_config.py \
  -n 10 -o out/config.yaml --use-default-catalog
```

Uses the built-in cloud-like instance table (multiple CPU/GPU shapes).

## Option 2: Command-line resource shape (recommended for custom single shapes)

```bash
# CPU and memory (required together for simple catalog)
python experiments/scripts/step3_system_config.py -n 10 -o out/config.yaml \
  --cpu 16 --memory 64

# Add GPUs
python experiments/scripts/step3_system_config.py -n 10 -o out/config.yaml \
  --cpu 32 --memory 128 --gpu 4 --gpu-memory 32

# All optional knobs
python experiments/scripts/step3_system_config.py -n 10 -o out/config.yaml \
  --cpu 32 --memory 128 --gpu 4 --gpu-memory 32 --storage 2000
```

**Parameters**

- `--cpu` — cores per instance (use with `--memory`)
- `--memory` — RAM (GB) per instance
- `--gpu` — GPU count (default `0`)
- `--gpu-memory` — GPU memory (GB)
- `--storage` — storage (GB); default scales with memory if omitted
- `-n` — number of samples

**Behavior:** the generator typically builds a **small / base / large** trio around your base shape, then LHS-samples `k` (instances) within `[k_min, k_max]`.

## Option 3: Custom resource-catalog YAML

```bash
python experiments/scripts/step3_system_config.py \
  -n 20 -o out/config.yaml --resource-catalog my_resources.yaml
```

**Example `my_resources.yaml`**

```yaml
resources:
  - cpu_cores: 8
    memory_gb: 32
    storage_gb: 200
    num_gpus: 0

  - cpu_cores: 16
    memory_gb: 64
    storage_gb: 500
    num_gpus: 1
    gpu_memory_gb: 16

  - cpu_cores: 32
    memory_gb: 128
    storage_gb: 1000
    num_gpus: 4
    gpu_memory_gb: 32
```

## Worked examples

### CPU-only cluster

```bash
python experiments/scripts/step3_system_config.py -n 10 -o out/config_cpu.yaml \
  --cpu 16 --memory 64
```

### GPU server

```bash
python experiments/scripts/step3_system_config.py -n 20 -o out/config_gpu.yaml \
  --cpu 32 --memory 128 --gpu 4 --gpu-memory 32
```

### End-to-end with merge

```bash
autoconfig query --input data/queries/kernel_bfs.cu -o out/query.yaml
autoconfig graph --input data/graph_medium_pl.csv -o out/graph.yaml
python experiments/scripts/step3_system_config.py -n 10 -o out/config.yaml \
  --cpu 32 --memory 128 --gpu 4 --gpu-memory 32
autoconfig merge -q out/query.yaml -g out/graph.yaml -c out/config.yaml -o out/merged.yaml
```

## Output format (sketch)

```yaml
configurations:
  - config_id: 0
    k: 6
    resource:
      cpu_cores: 32
      memory_gb: 128
      storage_gb: 1000
      num_gpus: 4
      gpu_memory_gb: 32
config_features:
  - conf_k_instances: 6
    conf_total_cpu_cores: 192
    ...
metadata:
  num_samples: 10
  k_range: [1, 16]
  sampling_method: latin_hypercube
```

## FAQ

**What is `-n`?** Short for `--num-samples`: number of `(k, resource)` candidates.

**How do I bound `k`?** `--k-min` and `--k-max` on `step3_system_config.py`.

**Why three resource shapes with `--cpu`/`--memory`?** To let LHS explore scale around your baseline.

**Only one hardware class?** Put a **single** entry in `resources:` and pass `--resource-catalog` so only `k` varies.

## Command reference (`step3_system_config.py`)

```
-n, --num-samples       Number of configurations
-o, --output            Output YAML
--resource-catalog, -c  Catalog YAML
--use-default-catalog   Built-in catalog
--cpu / --memory        Simple catalog (use together)
--gpu, --gpu-memory, --storage
--k-min, --k-max        Instance count range
```

## Sample shell script

```bash
#!/usr/bin/env bash
OUT="results"
mkdir -p "$OUT"

python experiments/scripts/step3_system_config.py -n 10 -o "$OUT/config_cpu.yaml" \
  --cpu 16 --memory 64 --k-min 1 --k-max 8

python experiments/scripts/step3_system_config.py -n 15 -o "$OUT/config_gpu.yaml" \
  --cpu 32 --memory 128 --gpu 4 --gpu-memory 32 --k-min 1 --k-max 4

python experiments/scripts/step3_system_config.py -n 20 -o "$OUT/config_large.yaml" \
  --use-default-catalog --k-min 2 --k-max 16
```

## Capacity YAML under `data/conf/`

Files such as `system_capacity_small.yaml` document **cluster-wide limits** (total CPU, memory, GPUs, per-machine caps). They are useful when designing experiments or extending generators to respect capacity; see [data/conf/README.md](../data/conf/README.md).
