# Configuration YAML for merge

Step 4 (`FeatureMerger`) expects a YAML file that lists **system configuration**
candidates. The `autoconfig` CLI does **not** run config generation, but the repo
includes a **small LHS sampler** you can use to emit sample files.

## `FeatureMerger` input shape

At minimum, the file should contain:

- **`catalog`**: (optional) list of machine / resource *types* $s \in \mathcal{S}$ — useful for documentation.
- **`configurations`**: a list of entries, each with:
  - **`resource`**: per-instance fields such as `cpu_cores`, `memory_gb`, `storage_gb`, `num_gpus`, and optional `grid_size` / `block_size` for GPU launches.
  - **`k`**: replica count, **or** omit `k` and set **`node_id`** for a single-node row (treated as one instance, `k = 1` in the merger).
- **`config_features`**: (optional) if omitted, the merger **derives** a numeric row per `configurations` entry from `k` and `resource` in `autoconfig/utils/feature_merger.py`.

**Merged vector — configuration block (12 numbers):**  
`conf_memory_limit`, `conf_num_threads`, `conf_cache_size`, `conf_batch_size`, `conf_io_buffer_size`, `conf_num_workers`, `conf_timeout`, `conf_enable_index`, `conf_index_type`, `conf_compression_enabled`, **`conf_grid_size`**, **`conf_block_size`**.  
Resource fields `grid_size` / `block_size` map to the last two; use `0` when there is no GPU / no launch grid.

**Typical combined merge width: 8 + 12 + 17 + 12 = 49** (static + symbolic + graph + config).

---

## Sample configs via Latin Hypercube: `data/conf/build_ten_conf.py`

The script **Latin-hypercube samples** 8 unit dimensions, maps them to integer
ranges, and enforces **GPU vs CPU** consistency:

- If **`num_gpus == 0`** then **`grid_size == 0`** and **`block_size == 0`**.
- If **`num_gpus > 0`** then **both** `grid_size` and `block_size` are **positive**
  (and optional `gpu_memory_gb`, `gpu_sm_count` are set).

A tunable fraction of rows use the **GPU** branch (see `--gpu-fraction`); the rest
are CPU-only. Default hardware ranges (all overridable) include
`cpu_cores` 1–64, `memory_gb` 1–128, `grid`/`block` 1–256, etc.

```bash
# Write data/conf/conf_01.yaml … conf_10.yaml
python data/conf/build_ten_conf.py -n 10 --seed 42

# Custom ranges / count
python data/conf/build_ten_conf.py -n 20 --cpu-max 32 --mem-max 64 --grid-max 256 --gpu-fraction 0.4
```

Copy or symlink one of the generated files to your pipeline output name (e.g. `out/config_features.yaml`) before merge.

---

## Running merge

```bash
autoconfig merge -q out/query_features.yaml -g out/graph_features.yaml -c your_config.yaml -o out/merged_features.yaml
```

For full field lists and naming, see [feature_extraction.md](feature_extraction.md).
