# 实验脚本导入修复说明

## 修复的导入问题

### 1. 移除不存在的 `BayesRidgeModel` 导入
- **文件**: `exp1_feature_extraction.py`
- **修复**: 删除了对不存在的 `BayesRidgeModel` 的导入
- **原因**: `autoconfig.models` 只导出 `BayesianExecutionTimeModel`, `BayesianTimeModel`, `BayesianCostModel`

### 2. 修复相对导入路径错误
所有实验脚本中使用了 `from experiments.*` 的相对导入，这在独立运行时会失败。

#### 文件: `exp1_feature_extraction.py`
```python
# 修复前（错误）
from experiments.workloads import query_wcc, query_sssp, query_pr, query_bfs, query_subiso

# 修复后（正确）
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))
from workloads import query_wcc, query_sssp, query_pr, query_bfs, query_subiso
```

#### 文件: `exp2_effectiveness.py`
```python
# 修复前（错误）
from experiments.baselines import (BayesianOptimization, ...)
from experiments.workloads import query_wcc, ...
from .exp1_feature_extraction import generate_training_data, ...

# 修复后（正确）
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))
from baselines import (BayesianOptimization, ...)
from workloads import query_wcc, ...

# 使用动态导入exp1工具函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp1_utils",
    Path(__file__).parent / "exp1_feature_extraction.py"
)
exp1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1_module)
```

#### 文件: `exp3_robustness.py`
```python
# 修复前（错误）
from experiments.baselines import (BayesianOptimization, ...)
from experiments.workloads import query_wcc, ...
from .exp1_feature_extraction import generate_training_data, ...

# 修复后（正确）
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))
from baselines import (BayesianOptimization, ...)
from workloads import query_wcc, ...

# 使用动态导入exp1工具函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp1_utils",
    Path(__file__).parent / "exp1_feature_extraction.py"
)
exp1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1_module)
```

#### 文件: `exp5_scalability.py`
```python
# 修复前（错误）
from experiments.workloads import query_wcc, ...
from .exp1_feature_extraction import generate_training_data, ...

# 修复后（正确）
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))
from workloads import query_wcc, ...

# 使用动态导入exp1工具函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp1_utils",
    Path(__file__).parent / "exp1_feature_extraction.py"
)
exp1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1_module)
```

#### 文件: `exp6_case_study.py`
```python
# 修复前（错误）
from experiments.baselines import (BayesianOptimization, ...)
from experiments.workloads import query_gar_match
from .exp1_feature_extraction import _compute_exec_time

# 修复后（正确）
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))
from baselines import (BayesianOptimization, ...)
from workloads import query_gar_match

# 使用动态导入exp1工具函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp1_utils",
    Path(__file__).parent / "exp1_feature_extraction.py"
)
exp1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1_module)
```

## 修复原理

### 为什么需要这些修复？

1. **`experiments` 不是已安装的包**
   - 它只是项目目录，不在Python路径中
   - 不能直接使用 `from experiments.*` 导入

2. **相对导入只在包中有意义**
   - 相对导入（如 `from .exp1_feature_extraction`）只适用于包内部
   - 脚本直接运行时不会成为包的一部分

3. **动态导入的优势**
   - 避免相对导入问题
   - 跨脚本共享工具函数（如 `generate_training_data`, `_compute_exec_time`）

### 修复策略

#### 策略1: 添加模块到路径
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
from baselines import BayesianOptimization  # 现在可以使用
```

#### 策略2: 动态导入
```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "module_name",
    Path(__file__).parent / "target_script.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 使用导入的模块
module.some_function()
```

## 如何验证修复

运行任意实验脚本，应该不再出现导入错误：

```bash
cd /Users/hsiaoko/Projects/AutoConfig
/opt/anaconda3/bin/python experiments/scripts/run_single_experiment.py --exp 1
```

或者使用提供的测试脚本：

```bash
cd /Users/hsiaoko/Projects/AutoConfig
chmod +x run_exp_test.sh
./run_exp_test.sh
```

## 已修复的文件列表

✅ `experiments/scripts/exp1_feature_extraction.py`
✅ `experiments/scripts/exp2_effectiveness.py`
✅ `experiments/scripts/exp3_robustness.py`
✅ `experiments/scripts/exp5_scalability.py`
✅ `experiments/scripts/exp6_case_study.py`

## 测试运行

修复后的导入应该可以正常工作。如果仍有问题，请检查：

1. Python环境是否正确
2. 所有依赖包是否已安装
3. `sys.path` 是否包含正确的路径

生成的日志会显示具体的错误信息，便于进一步诊断。