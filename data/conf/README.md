# AutoConfig 配置生成示例

本目录包含系统容量配置示例和完整的使用演示。

## 系统容量配置文件

### small (小型系统)

```yaml
# data/conf/system_capacity_small.yaml
total_cpu_cores: 64
total_memory_gb: 256
total_storage_gb: 2000
total_gpus: 4
total_gpu_memory_gb: 128

machine:
  cpu_cores: 16
  memory_gb: 64
  storage_gb: 500
  num_gpus: 1
  gpu_memory_gb: 32
```

**最大机器数**: 4 台 (受限于 CPU: 64/16=4, 内存：256/64=4, GPU: 4/1=4)

### medium (中型集群)

```yaml
# data/conf/system_capacity_medium.yaml
total_cpu_cores: 256
total_memory_gb: 1024
total_storage_gb: 10000
total_gpus: 16
total_gpu_memory_gb: 512

machine:
  cpu_cores: 32
  memory_gb: 128
  storage_gb: 1000
  num_gpus: 2
  gpu_memory_gb: 32
```

**最大机器数**: 8 台

### large (大型 GPU 集群)

```yaml
# data/conf/system_capacity_large.yaml
total_cpu_cores: 512
total_memory_gb: 2048
total_storage_gb: 20000
total_gpus: 32
total_gpu_memory_gb: 1024

machine:
  cpu_cores: 64
  memory_gb: 256
  storage_gb: 2000
  num_gpus: 4
  gpu_memory_gb: 32
```

**最大机器数**: 8 台

### cpu_only (纯 CPU 系统)

```yaml
# data/conf/system_capacity_cpu_only.yaml
total_cpu_cores: 128
total_memory_gb: 512
total_storage_gb: 5000
total_gpus: 0
total_gpu_memory_gb: 0

machine:
  cpu_cores: 32
  memory_gb: 128
  storage_gb: 1000
  num_gpus: 0
```

**最大机器数**: 4 台

---

## 使用示例

### 示例 1: 基本使用

```bash
# 使用小型系统容量生成 5 个配置
autoconfig config \
    --capacity data/conf/system_capacity_small.yaml \
    -n 5 \
    -o out/configs.yaml
```

**输出**:
- 5 个配置样本
- k 范围：[1, 4] (受容量限制)
- 每个配置包含 k 台机器的资源

### 示例 2: 完整流程

```bash
# 运行完整示例脚本
bash examples/run_full_pipeline.sh
```

这会执行：
1. 提取查询特征 (GAR Match CUDA 代码)
2. 提取图特征 (Power-Law 图)
3. 生成容量受限的配置
4. 合并所有特征

### 示例 3: 自定义 k 范围

```bash
# 指定 k 的范围
autoconfig config \
    --capacity data/conf/system_capacity_medium.yaml \
    -n 10 \
    --k-min 2 \
    --k-max 6 \
    -o out/configs.yaml
```

**输出**:
- 10 个配置样本
- k 范围：[2, 6] (在容量范围内)

---

## 输出文件格式

```yaml
system_capacity:
  total_cpu_cores: 64
  total_memory_gb: 256
  total_gpus: 4
  machine:
    cpu_cores: 16
    memory_gb: 64
    num_gpus: 1

configurations:
  - config_id: 0
    k: 1
    resource:
      cpu_cores: 16
      memory_gb: 64
      num_gpus: 1
      gpu_memory_gb: 32
  
  - config_id: 1
    k: 3
    resource:
      cpu_cores: 48
      memory_gb: 192
      num_gpus: 3
      gpu_memory_gb: 96

config_features:
  - conf_k_instances: 1
    conf_total_cpu_cores: 16
    conf_total_memory_gb: 64
    conf_total_gpus: 1
    conf_per_instance:
      cpu_cores: 16
      memory_gb: 64
      num_gpus: 1
    ...

metadata:
  num_samples: 5
  k_range: [1, 4]
  max_machines: 4
  sampling_method: latin_hypercube_capacity_constrained
```

---

## 工作原理

### 1. 系统容量定义

```yaml
total_cpu_cores: 64      # 总 CPU 核心数
total_memory_gb: 256     # 总内存
total_gpus: 4            # 总 GPU 数
machine:                 # 单机配置
  cpu_cores: 16
  memory_gb: 64
  num_gpus: 1
```

### 2. 计算最大机器数

```
max_machines = min(
    total_cpu_cores / machine.cpu_cores,      # 64/16 = 4
    total_memory_gb / machine.memory_gb,      # 256/64 = 4
    total_gpus / machine.num_gpus             # 4/1 = 4
) = 4
```

### 3. LHS 采样

在 k ∈ [1, max_machines] 范围内使用拉丁超立方采样生成 k 值。

### 4. 生成配置

对于每个 k 值：
```python
resource = {
    'cpu_cores': machine.cpu_cores * k,
    'memory_gb': machine.memory_gb * k,
    'num_gpus': machine.num_gpus * k,
    'gpu_memory_gb': machine.gpu_memory_gb * k,
}
```

---

## 对比：容量约束 vs 自由配置

### 容量约束模式 (推荐)

```bash
autoconfig config \
    --capacity data/conf/system_capacity_small.yaml \
    -n 10 -o out/configs.yaml
```

**特点**:
- 配置受系统总容量限制
- k 的范围自动计算
- 适合真实集群环境

### 自由配置模式

```bash
autoconfig config \
    --cpu 16 --memory 64 \
    -n 10 -o out/configs.yaml
```

**特点**:
- 生成 3 种资源变体 (小/中/大)
- k 范围需要手动指定
- 适合探索性实验

---

## 运行完整示例

```bash
cd autoconfig
source venv/bin/activate

# 运行完整流程
bash examples/run_full_pipeline.sh

# 查看输出
ls -lh out/example/
cat out/example/merged_features.yaml
```

输出目录结构：
```
out/example/
├── query_features.yaml    # 查询特征
├── graph_features.yaml    # 图特征
├── config_features.yaml   # 配置特征
└── merged_features.yaml   # 合并后的特征矩阵
```

---

## 常见问题

### Q: 如何自定义机器配置？

**A**: 修改 capacity YAML 中的 `machine` 部分：

```yaml
machine:
  cpu_cores: 64      # 每台机器 64 核
  memory_gb: 256     # 每台机器 256GB 内存
  num_gpus: 8        # 每台机器 8 块 GPU
  gpu_memory_gb: 80  # 每块 GPU 80GB 显存
```

### Q: 如何指定 k 的最大值？

**A**: 使用 `--k-max` 参数（不能超过容量限制）：

```bash
autoconfig config \
    --capacity system.yaml \
    -n 10 \
    --k-max 6  # 最多 6 台机器
```

### Q: 如果没有 GPU 怎么办？

**A**: 使用 `system_capacity_cpu_only.yaml` 或设置 `total_gpus: 0`：

```bash
autoconfig config \
    --capacity data/conf/system_capacity_cpu_only.yaml \
    -n 10 -o out/configs.yaml
```

### Q: 如何增加样本数量？

**A**: 增加 `-n` 参数：

```bash
autoconfig config \
    --capacity system.yaml \
    -n 50  # 生成 50 个配置样本
    -o out/configs.yaml
```
