# 配置生成工具使用说明

## 问题说明

之前的命令：
```bash
autoconfig config -n 10 -o out/config.yaml --use-default-catalog
```

**问题**：
- `-n 10` 只是生成 10 个配置样本
- 但资源配置来自**内置的默认目录**（11 种云实例）
- 用户无法指定自己的 CPU、GPU、内存等参数

## 解决方案

现在有三种方式指定资源配置：

### 方式 1: 使用默认目录（快速测试）

```bash
autoconfig config -n 10 -o out/config.yaml --use-default-catalog
```

使用内置的 11 种云实例配置（从 2 CPU 到 64 CPU，含 GPU 实例）。

### 方式 2: 命令行指定资源参数（推荐）

```bash
# 指定 CPU 和内存
autoconfig config -n 10 -o out/config.yaml --cpu 16 --memory 64

# 指定 CPU、内存和 GPU
autoconfig config -n 10 -o out/config.yaml \
    --cpu 32 --memory 128 --gpu 4 --gpu-memory 32

# 指定所有参数
autoconfig config -n 10 -o out/config.yaml \
    --cpu 32 --memory 128 --gpu 4 --gpu-memory 32 --storage 2000
```

**参数说明**：
- `--cpu <cores>`: CPU 核心数（必需，与 --memory 一起使用）
- `--memory <GB>`: 内存大小（必需）
- `--gpu <num>`: GPU 数量（可选，默认 0）
- `--gpu-memory <GB>`: GPU 显存（可选）
- `--storage <GB>`: 存储空间（可选，默认 10×内存）
- `-n <num>`: 生成的配置样本数量（默认 20）

**工作原理**：
根据你指定的参数，自动生成 3 种资源变体：
1. **小配置**: 1/2 资源
2. **基础配置**: 你指定的资源
3. **大配置**: 2× 资源

然后使用 LHS 采样生成 k 个实例的组合。

### 方式 3: 自定义资源目录 YAML

```bash
autoconfig config -n 20 -o out/config.yaml --resource-catalog my_resources.yaml
```

**my_resources.yaml 示例**：
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

---

## 使用示例

### 示例 1: 小型集群（无 GPU）

```bash
autoconfig config -n 10 -o out/config.yaml \
    --cpu 16 --memory 64
```

**输出**：
- 3 种资源变体：(8C/32G, 16C/64G, 32C/128G)
- 10 个配置样本：k 从 1-16 变化
- 总共 10 个 (k, resource) 组合

### 示例 2: GPU 服务器

```bash
autoconfig config -n 20 -o out/config.yaml \
    --cpu 32 --memory 128 --gpu 4 --gpu-memory 32
```

**输出**：
- 3 种资源变体：
  - 小：16C/64G/3GPU/16G
  - 基础：32C/128G/4GPU/32G
  - 大：64C/256G/8GPU/64G
- 20 个配置样本

### 示例 3: 完整流程

```bash
# 1. 提取查询特征
autoconfig query --input queries/gar_match.cu -o out/query.yaml

# 2. 提取图特征
autoconfig graph --input data/graph_medium_pl.csv -o out/graph.yaml

# 3. 生成配置（指定资源）
autoconfig config -n 10 -o out/config.yaml \
    --cpu 32 --memory 128 --gpu 4 --gpu-memory 32

# 4. 合并特征
autoconfig merge -q out/query.yaml -g out/graph.yaml -c out/config.yaml \
    -o out/merged.yaml
```

---

## 输出文件格式

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
      gpu_sm_count: 0
  
  - config_id: 1
    k: 7
    resource:
      cpu_cores: 32
      memory_gb: 128
      ...

config_features:
  - conf_k_instances: 6
    conf_total_cpu_cores: 192
    conf_total_memory_gb: 768
    conf_total_storage_gb: 6000
    conf_total_gpus: 24
    conf_per_instance:
      cpu_cores: 32
      memory_gb: 128
      ...

metadata:
  num_samples: 10
  k_range: [1, 16]
  catalog_size: 3  # 3 种资源变体
  sampling_method: latin_hypercube
```

---

## 常见问题

### Q: `-n` 是什么？

**A**: `-n` 是 `--num-samples` 的缩写，表示**生成的配置样本数量**。

例如 `-n 10` 会生成 10 个不同的配置（每个配置是 k 实例数 + 资源类型的组合）。

### Q: 如何指定 k 的范围？

**A**: 使用 `--k-min` 和 `--k-max`：

```bash
autoconfig config -n 10 -o out/config.yaml \
    --cpu 16 --memory 64 \
    --k-min 2 --k-max 8
```

这会在 k ∈ [2, 8] 范围内采样。

### Q: 为什么输出有 3 种资源变体？

**A**: 当你使用 `--cpu --memory` 时，系统自动生成 3 种变体：
- 小配置（1/2 资源）
- 基础配置（你指定的）
- 大配置（2× 资源）

这样 LHS 可以在不同规模之间采样，获得更好的覆盖。

### Q: 如何只生成一种资源配置？

**A**: 使用 YAML 目录文件：

```yaml
# single_resource.yaml
resources:
  - cpu_cores: 32
    memory_gb: 128
    storage_gb: 1000
    num_gpus: 4
    gpu_memory_gb: 32
```

```bash
autoconfig config -n 10 -c single_resource.yaml -o out/config.yaml
```

这样只会在这一种资源上变化 k 值。

---

## 命令参考

```bash
autoconfig config [选项]

选项:
  -n, --num-samples NUM      生成的配置样本数 (默认：20)
  -o, --output FILE          输出 YAML 文件 (默认：out/config_features.yaml)
  
  # 资源指定（三选一）:
  -c, --resource-catalog FILE  资源目录 YAML 文件
  --use-default-catalog        使用默认云资源目录
  --cpu CORES --memory GB      自定义资源 (必需一起使用)
      --gpu NUM                GPU 数量 (默认：0)
      --gpu-memory GB          GPU 显存 (默认：0)
      --storage GB             存储空间 (默认：10×内存)
  
  # LHS 参数:
  --k-min NUM                  最小实例数 (默认：1)
  --k-max NUM                  最大实例数 (默认：16)
```

---

## 完整示例脚本

```bash
#!/bin/bash
# generate_configs.sh

OUTPUT_DIR="results"
mkdir -p $OUTPUT_DIR

echo "=== Generating Configurations ==="

# Scenario 1: CPU-only cluster
echo "Scenario 1: CPU-only (16 cores, 64GB)"
autoconfig config -n 10 -o $OUTPUT_DIR/config_cpu.yaml \
    --cpu 16 --memory 64 \
    --k-min 1 --k-max 8

# Scenario 2: GPU server
echo "Scenario 2: GPU server (32 cores, 128GB, 4 GPUs)"
autoconfig config -n 15 -o $OUTPUT_DIR/config_gpu.yaml \
    --cpu 32 --memory 128 --gpu 4 --gpu-memory 32 \
    --k-min 1 --k-max 4

# Scenario 3: Large cluster (using default catalog)
echo "Scenario 3: Large cluster (default catalog)"
autoconfig config -n 20 -o $OUTPUT_DIR/config_large.yaml \
    --use-default-catalog \
    --k-min 2 --k-max 16

echo "=== Done ==="
ls -lh $OUTPUT_DIR/
```
