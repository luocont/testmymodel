# MagNet 快速配置指南

## 配置概述

当前配置：
- **多智能体框架**：阿里云百炼（qwen2.5-7b-instruct）
- **评估系统**：OpenRouter（openai/gpt-4o）

## 步骤 1: 配置 API 密钥

编辑项目根目录的 `.env` 文件：

```bash
# 多智能体框架配置（阿里云百炼）
LLM_PROVIDER=aliyun
LLM_API_KEY=sk-your-dashscope-api-key-here
LLM_MODEL=qwen2.5-7b-instruct

# 评估系统配置（OpenRouter）
EVAL_LLM_PROVIDER=openrouter
EVAL_LLM_API_KEY=sk-or-your-openrouter-api-key-here
EVAL_LLM_MODEL=openai/gpt-4o
```

### 获取 API 密钥

**阿里云百炼：**
1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 创建 API Key
3. 将密钥填入 `LLM_API_KEY`

**OpenRouter：**
1. 访问 [OpenRouter](https://openrouter.ai/)
2. 创建账户并获取 API Key
3. 将密钥填入 `EVAL_LLM_API_KEY`

## 步骤 2: 加载环境变量

### Linux/Mac
```bash
source .env
```

### Windows PowerShell
```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
```

### Windows CMD
```cmd
# 手动设置每个环境变量
set LLM_PROVIDER=aliyun
set LLM_API_KEY=your_key
set LLM_MODEL=qwen2.5-7b-instruct
set EVAL_LLM_PROVIDER=openrouter
set EVAL_LLM_API_KEY=your_key
set EVAL_LLM_MODEL=openai/gpt-4o
```

## 步骤 3: 运行脚本

### 运行多智能体框架生成对话
```bash
cd src
python inference-parallel-magnet.py -o ../output -num_pr 4 -m_turns 20
```

### 运行评估系统

**CTRS 评估：**
```bash
cd evaluation/CTRS
python ctrs-gpt4o.py -i ../../output -o ../../results/CTRS
```

**WAI 评估：**
```bash
cd evaluation/WAI
python wai-gpt4o.py -i ../../output -o ../../results/WAI
```

**PANAS 评估（咨询前）：**
```bash
cd evaluation/PANAS
python panas_before-gpt4o.py -o ../../results/PANAS/before
```

**PANAS 评估（咨询后）：**
```bash
cd evaluation/PANAS
python panas_after-gpt4o.py -i ../../output -o ../../results/PANAS/after
```

## 验证配置

运行前检查输出是否显示：

**多智能体框架：**
```
从环境变量加载 LLM 配置: aliyun
使用模型: qwen2.5-7b-instruct
```

**评估系统：**
```
从环境变量加载评估 API 配置: openrouter
使用评估模型: openai/gpt-4o
```

## 成本估算

| 组件 | 模型 | 用途 | 参考成本 |
|------|------|------|----------|
| 多智能体 | qwen2.5-7b-instruct | 对话生成 | 约 ¥0.004/千 tokens |
| 技术智能体 | qwen-max | 技术选择 | 约 ¥0.02/千 tokens |
| 评估系统 | gpt-4o | 质量评估 | 约 $0.005/千 tokens |

## 常见问题

### Q: 如何更换模型？

**A:** 修改 `.env` 文件中的 `LLM_MODEL` 或 `EVAL_LLM_MODEL`：

```bash
# 使用其他阿里云模型
LLM_MODEL=qwen-plus

# 使用 OpenRouter 上的其他模型
EVAL_LLM_MODEL=anthropic/claude-3.5-sonnet
```

### Q: 如何全部使用阿里云？

**A:** 修改 `.env` 文件：
```bash
LLM_PROVIDER=aliyun
LLM_API_KEY=your_dashscope_key
LLM_MODEL=qwen2.5-7b-instruct

EVAL_LLM_PROVIDER=aliyun
EVAL_LLM_API_KEY=your_dashscope_key
EVAL_LLM_MODEL=qwen-max
```

### Q: 环境变量未生效？

**A:** 确保在运行脚本前加载了环境变量：
```bash
# 检查环境变量
echo $LLM_PROVIDER
echo $LLM_API_KEY
```

### Q: API 调用失败？

**A:** 检查：
1. API 密钥是否正确
2. 网络连接是否正常
3. API 额度是否充足
4. 模型名称是否正确

## 技术支持

- 详细配置指南：[docs/API_CONFIG_GUIDE.md](API_CONFIG_GUIDE.md)
- 代码示例：[examples/api_usage_examples.py](../examples/api_usage_examples.py)
- 问题反馈：[GitHub Issues](https://github.com/your-repo/issues)
