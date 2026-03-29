# Development Harness - 快速开始

## 安装

```bash
cd /Users/hsiaoko/Projects/AutoConfig

# 安装依赖
pip install langchain langchain-openai langchain-community pyyaml
```

## 配置 API Key

```bash
# 方式 1：环境变量（推荐）
export OPENAI_API_KEY=sk-...

# 方式 2：编辑 config.yaml（不推荐用于生产环境）
# 在 dev_harness/config.yaml 中添加 api_keys 部分
```

## 使用

### 1. 查看状态

```bash
python -m dev_harness status
```

输出示例：
```
=== Development Harness Status ===
Project Root: /Users/hsiaoko/Projects/AutoConfig
LLM Model: gpt-4o-mini
Available Agents: dev, test, doc, review
Verbose: false
```

### 2. 列出 Agent

```bash
python -m dev_harness list-agents
```

输出：
```
Available Agents:
  - dev
  - test
  - doc
  - review
```

### 3. 运行开发任务

```bash
# 开发新功能
python -m dev_harness run --task "为 config_generator.py 添加 GPU 资源支持"

# 指定目标文件
python -m dev_harness run --target autoconfig/utils/config_generator.py --task "添加最大实例数限制参数"
```

### 4. 运行测试 Agent

```bash
# 为特定文件生成测试
python -m dev_harness run --agent test --target autoconfig/utils/config_generator.py

# 运行测试并验证
python -m dev_harness run --agent test --task "为 CapacityConstrainedConfigGenerator 写单元测试"
```

### 5. 运行文档 Agent

```bash
# 生成 API 文档
python -m dev_harness run --agent doc --task "为 cli.py 生成 API 文档"
```

### 6. 运行审查 Agent

```bash
# 代码审查
python -m dev_harness run --agent review --target autoconfig/cli.py
```

### 7. 运行完整管道

```bash
# 完整流程：开发→测试→审查→文档
python -m dev_harness pipeline --task "实现 GPU 资源配置功能"

# 自定义工作流
python -m dev_harness pipeline --task "修复 query_extractor 的 bug" --workflow dev,test,review
```

## 配置文件

编辑 `dev_harness/config.yaml`：

```yaml
llm:
  provider: openai
  model: gpt-4o-mini        # 或 gpt-4o, gpt-3.5-turbo
  temperature: 0.2          # 0.0-1.0，越低越确定
  max_tokens: 4096

agents:
  verbose: false            # 详细输出

pipeline:
  max_iterations: 10        # 每 Agent 最大迭代次数
  timeout: 300              # 超时（秒）
```

## 日志

日志保存在 `.harness-logs/` 目录：

```bash
# 查看最新日志
tail -f .harness-logs/harness_*.log
```

## 常见问题

### Q: 如何更改使用的模型？

编辑 `dev_harness/config.yaml`：
```yaml
llm:
  model: gpt-4o  # 或 gpt-4o-mini, gpt-3.5-turbo
```

### Q: 如何增加 verbosity？

```bash
python -m dev_harness run --task "..." --verbose
```

或在 `config.yaml` 中：
```yaml
agents:
  verbose: true
```

### Q: 任务执行超时怎么办？

增加超时时间：
```yaml
pipeline:
  timeout: 600  # 增加到 600 秒
```

### Q: 如何查看可用的工具？

每个 Agent 都可以使用以下工具：
- `read_file` - 读取文件
- `write_file` - 写入文件
- `list_directory` - 列出目录
- `search_code` - 搜索代码
- `run_tests` - 运行测试
- `check_syntax` - 检查语法

## 下一步

- 查看完整文档：`dev_harness/README.md`
- 尝试运行第一个任务
- 根据你的需求调整配置
