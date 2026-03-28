# Query Code Examples

本目录包含图查询代码示例，用于特征提取测试。

## 查询文件

### 1. GAR Match (CUDA)

**文件**: `gar_match.cu`

**说明**: 图属性规则（GAR）匹配 CUDA 内核实现

**特征**:
- 多阶段 CUDA 内核执行
- 顶点过滤和边过滤
- 候选生成和扩展
- 原子操作（atomicAdd）
- 设备同步（cudaDeviceSynchronize）

**使用**:
```bash
autoconfig query --input data/queries/gar_match.cu -o out/query.yaml
```

**预期特征**:
- 静态特征：8 个（循环、分支、原子操作、同步等）
- 符号特征：12 个（VScan、EScan、Atom 等）

### 2. BFS (Python)

**文件**: `query_bfs.py`

**说明**: 广度优先搜索算法实现

**特征**:
- 前沿驱动迭代（frontier-driven）
- 边扫描模式
- 无递归

**使用**:
```bash
autoconfig query --input data/queries/query_bfs.py -o out/query.yaml
```

### 3. GAR Match (Simplified)

**文件**: `query_gar_match.cu`

**说明**: GAR Match 的简化版本，用于快速测试

**特征**: 与 gar_match.cu 类似，但代码更简洁

---

## 添加新查询

将查询代码文件放入此目录，然后使用：

```bash
autoconfig query --input data/queries/your_query.cu -o out/query.yaml
```

## 特征提取说明

### 静态特征（8 个）

从代码结构提取，与输入图无关：
1. 循环计数
2. 最大循环深度
3. 分支计数
4. 变量计数
5. 递归计数
6. 原子操作计数
7. 同步操作计数
8. 显式并行标志

### 符号特征（12 个）

识别代码模式，使用占位符（在 merge 阶段实例化）：
1. VScan - 顶点扫描
2. EScan - 边扫描
3. FScan - 前沿迭代
4. RExp - 递归扩展
5. Atom - 原子更新
6. Comm - 跨分区通信

每个特征包含：
- `coeff`: 系数（占位符 1.0，merge 时实例化）
- `requires`: 是否需要图统计（1.0 = 需要）

---

## 示例输出

```yaml
query_features:
  static:
    static_loop_count: 8.0
    static_max_loop_depth: 2.0
    static_branch_count: 16.0
    static_variable_count: 53.0
    static_recursion_count: 0.0
    static_atomic_op_count: 4.0
    static_sync_count: 7.0
    static_explicit_parallel_flag: 1.0
  symbolic:
    sym_vscan_coeff: 1.0
    sym_vscan_requires: 1.0
    sym_escan_coeff: 1.0
    sym_escan_requires: 1.0
    sym_atom_coeff: 1.0
    sym_atom_requires: 1.0
    ...
```

---

## 相关文件

- `data/graph_*.csv` - 图数据文件
- `data/conf/system_capacity_*.yaml` - 系统容量配置
- `out/query_features.yaml` - 提取的查询特征输出
