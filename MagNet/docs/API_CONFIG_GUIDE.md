# MagNet API 配置指南

本文档详细说明如何为 MagNet 多智能体框架和评估系统配置不同的 API 提供商。

## 目录

- [支持的 API 提供商](#支持的-api-提供商)
- [快速开始](#快速开始)
- [配置方式](#配置方式)
- [API 提供商配置详情](#api-提供商配置详情)
- [代码示例](#代码示例)
- [常见问题](#常见问题)

---

## 支持的 API 提供商

| 提供商 | 多智能体框架 | 评估系统 | 推荐模型 |
|--------|-------------|---------|---------|
| **阿里云百炼** | ✅ | ✅ | qwen2.5-7b-instruct, qwen-max |
| **OpenRouter** | ✅ | ✅ | openai/chatgpt-4o-latest |
| **本地 vLLM** | ✅ | ❌ | 任意开源模型 |
| **Azure OpenAI** | ✅ | ✅ | gpt-4o, gpt-4o-mini |

---

## 快速开始

### 方式一：使用环境变量（推荐）

1. 复制配置模板：
```bash
cp config_template.env .env
```

2. 编辑 `.env` 文件，填写您的 API 配置

3. 在运行脚本前加载环境变量：
```bash
# Linux/Mac
source .env

# Windows PowerShell
Get-Content .env | ForEach-Object { $var = $_.Split('='); [System.Environment]::SetEnvironmentVariable($var[0], $var[1]) }

# 或直接设置
export LLM_PROVIDER=aliyun
export LLM_API_KEY=your_api_key
```

### 方式二：代码中直接配置

在 Python 脚本中直接配置（详见[代码示例](#代码示例)）。

---

## 配置方式

### 多智能体框架配置

多智能体框架支持三种配置方式：

#### 1. 从环境变量自动加载

```python
from llm_client import create_client_from_env

client = create_client_from_env()
```

环境变量：
- `LLM_PROVIDER`: 提供商 (aliyun/openrouter/local/azure)
- `LLM_API_KEY`: API 密钥
- `LLM_BASE_URL`: API 基础 URL (可选)
- `LLM_MODEL`: 模型名称
- `LLM_TEMPERATURE`: 温度参数
- `LLM_MAX_TOKENS`: 最大 token 数

#### 2. 使用便捷函数

```python
from llm_client import create_aliyun_client, create_openrouter_client

# 阿里云百炼
client = create_aliyun_client(
    api_key="your_api_key",
    model="qwen2.5-7b-instruct"
)

# OpenRouter
client = create_openrouter_client(
    api_key="your_api_key",
    model="openai/chatgpt-4o-latest"
)
```

#### 3. 使用 APIConfig 自定义

```python
from llm_client import LLMClient, APIConfig, APIProvider

config = APIConfig(
    provider=APIProvider.ALIYUN,
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen2.5-7b-instruct",
    temperature=0.7,
    max_tokens=512
)
client = LLMClient(config)
```

### 评估系统配置

评估系统的配置方式与多智能体框架类似：

#### 1. 从环境变量自动加载

```python
from llm_evaluator import create_eval_client_from_env

evaluator = create_eval_client_from_env()
```

环境变量：
- `EVAL_LLM_PROVIDER`: 提供商 (aliyun/openrouter/azure)
- `EVAL_LLM_API_KEY`: API 密钥
- `EVAL_LLM_BASE_URL`: API 基础 URL (可选)
- `EVAL_LLM_MODEL`: 模型名称
- `EVAL_LLM_TEMPERATURE`: 温度参数
- `EVAL_LLM_MAX_TOKENS`: 最大 token 数

#### 2. 使用便捷函数

```python
from llm_evaluator import create_eval_aliyun_client, create_eval_openrouter_client

# 阿里云百炼
evaluator = create_eval_aliyun_client(
    api_key="your_api_key",
    model="qwen-max"
)

# OpenRouter
evaluator = create_eval_openrouter_client(
    api_key="your_api_key",
    model="openai/gpt-4o"
)
```

---

## API 提供商配置详情

### 阿里云百炼

**获取 API Key：**
1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 创建 API Key
3. 设置环境变量 `DASHSCOPE_API_KEY`

**推荐模型：**
- 多智能体框架: `qwen2.5-7b-instruct` 或 `qwen-plus`
- 评估系统: `qwen-max`

**配置示例：**
```bash
# 环境变量方式
export LLM_PROVIDER=aliyun
export LLM_API_KEY=sk-xxxxxxxxxxxxx
export LLM_MODEL=qwen2.5-7b-instruct

# 或使用便捷函数
client = create_aliyun_client(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen2.5-7b-instruct"
)
```

**成本参考：**
- qwen2.5-7b-instruct: 约 ¥0.004/千 tokens
- qwen-max: 约 ¥0.02/千 tokens

### OpenRouter

**获取 API Key：**
1. 访问 [OpenRouter](https://openrouter.ai/)
2. 创建账户并获取 API Key
3. 设置环境变量 `OPENROUTER_API_KEY`

**推荐模型：**
- 多智能体框架: `openai/chatgpt-4o-latest` 或 `anthropic/claude-3.5-sonnet`
- 评估系统: `openai/gpt-4o`

**配置示例：**
```bash
# 环境变量方式
export LLM_PROVIDER=openrouter
export LLM_API_KEY=sk-or-xxxxxxxxxxxxx
export LLM_MODEL=openai/chatgpt-4o-latest

# 或使用便捷函数
client = create_openrouter_client(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/chatgpt-4o-latest"
)
```

**注意：** OpenRouter 按各模型提供商的定价计费，详见 [OpenRouter 定价](https://openrouter.ai/models)。

### 本地 vLLM

**部署步骤：**
1. 安装 vLLM: `pip install vllm`
2. 启动 vLLM 服务器:
```bash
vllm serve model_name --port 8000
```

**配置示例：**
```bash
# 环境变量方式
export LLM_PROVIDER=local
export LLM_API_KEY=dummy-key
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=model_name

# 或使用便捷函数
client = create_local_client(
    base_url="http://localhost:8000/v1",
    model="model_name"
)
```

**优势：** 无需 API 费用，适合有 GPU 资源的用户。

### Azure OpenAI

**获取 API Key：**
1. 在 Azure Portal 创建 OpenAI 资源
2. 获取密钥和端点 URL
3. 部署所需的模型

**配置示例：**
```bash
# 环境变量方式
export LLM_PROVIDER=azure
export LLM_API_KEY=your_api_key
export LLM_AZURE_ENDPOINT=https://your-resource.openai.azure.com
export LLM_API_VERSION=2024-02-15-preview
export LLM_DEPLOYMENT=gpt-4o-mini

# 或使用便捷函数
client = create_azure_client(
    api_key="your_api_key",
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o-mini"
)
```

---

## 代码示例

### 示例 1: 全部使用阿里云百炼

```python
# 配置环境变量
os.environ["LLM_PROVIDER"] = "aliyun"
os.environ["LLM_API_KEY"] = "your_dashscope_key"
os.environ["LLM_MODEL"] = "qwen2.5-7b-instruct"

os.environ["EVAL_LLM_PROVIDER"] = "aliyun"
os.environ["EVAL_LLM_API_KEY"] = "your_dashscope_key"
os.environ["EVAL_LLM_MODEL"] = "qwen-max"

# 或在代码中配置
from llm_client import create_aliyun_client
from llm_evaluator import create_eval_aliyun_client

client = create_aliyun_client(
    api_key="your_dashscope_key",
    model="qwen2.5-7b-instruct"
)

evaluator = create_eval_aliyun_client(
    api_key="your_dashscope_key",
    model="qwen-max"
)
```

### 示例 2: 多智能体使用本地 vLLM，评估使用阿里云

```python
from llm_client import create_local_client
from llm_evaluator import create_eval_aliyun_client

# 多智能体使用本地模型
client = create_local_client(
    base_url="http://localhost:8000/v1",
    model="model_name"
)

# 评估使用阿里云
evaluator = create_eval_aliyun_client(
    api_key="your_dashscope_key",
    model="qwen-max"
)
```

### 示例 3: 在 inference-parallel-magnet.py 中配置

```python
# 在文件顶部的 API 客户端配置区域
import os
from llm_client import create_aliyun_client

# 配置主客户端
client = create_aliyun_client(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen2.5-7b-instruct",
    temperature=0.7,
    max_tokens=512
)
# 注意：需要更新 generate() 函数以支持新的客户端

# 配置技术智能体客户端（可选）
technique_agent_client = create_aliyun_client(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max",  # 使用更强大的模型
    temperature=0
)
technique_agent_llm = technique_agent_client.client
```

### 示例 4: 在评估脚本中配置

```python
# 在评估脚本（如 ctrs-gpt4o.py）中
from llm_evaluator import create_eval_aliyun_client
import os

# 替换原有的 evaluator 配置
evaluator_client = create_eval_aliyun_client(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max",
    temperature=0,
    max_tokens=4096
)
evaluator = evaluator_client.client
model_for_cost = evaluator_client.model
```

---

## 常见问题

### Q: 如何切换不同的 API 提供商？

A: 有三种方式：
1. 修改环境变量 `LLM_PROVIDER` 和 `EVAL_LLM_PROVIDER`
2. 在代码中使用不同的便捷函数（如 `create_aliyun_client` vs `create_openrouter_client`）
3. 修改 `APIConfig` 中的 `provider` 参数

### Q: 可以为不同的智能体使用不同的 API 吗？

A: 可以。例如：
```python
# 客户端智能体使用阿里云
client_agent_client = create_aliyun_client(model="qwen2.5-7b-instruct")

# 技术智能体使用 OpenRouter
technique_agent_client = create_openrouter_client(model="openai/gpt-4o")
```

### Q: 如何估算成本？

A: 每个脚本都有 `calculate_cost()` 函数，运行后会输出成本信息。您可以参考各提供商的官方定价：
- [阿里云百炼定价](https://bailian.console.aliyun.com/)
- [OpenRouter 定价](https://openrouter.ai/models)

### Q: 本地 vLLM 可以用于评估吗？

A: 技术上可以，但不推荐。评估系统需要强大的模型来保证评分质量，建议使用云端的高质量模型（如 GPT-4o、qwen-max）。

### Q: 如何处理 API 限流？

A:
1. 使用重试机制（代码中已部分实现）
2. 降低并发数（减少 `num_processes`）
3. 分批处理数据

### Q: 兼容旧代码吗？

A: 完全兼容。新的配置模块不会影响现有的 Azure OpenAI 和本地 vLLM 配置。

---

## 技术支持

如有问题，请：
1. 查看 GitHub Issues
2. 参考官方文档
3. 检查 API 密钥和端点配置

---

**最后更新:** 2026-02-04
