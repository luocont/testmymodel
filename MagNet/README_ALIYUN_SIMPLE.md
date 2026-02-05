# MultiAgentESC + 阿里云 API 快速开始（简化版）

## 特点

✅ **简化版** - 不依赖 MultiAgentESC 的复杂多智能体系统
✅ **使用 MultiAgentESC 提示词** - 保留了核心的零样本提示词
✅ **无需 autogen** - 只依赖 OpenAI 兼容 API
✅ **完全独立** - 无需 MultiAgentESC 框架的其他部分

## 快速开始

### 1. 设置 API Key

```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-your-api-key-here"

# Windows CMD
set DASHSCOPE_API_KEY=sk-your-api-key-here
```

### 2. 运行

```bash
cd MagNet/src
python inference-multiagentesc-aliyun.py --model qwen2.5-7b-instruct --num_samples 10
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_key` | 环境变量 | 阿里云 API Key |
| `--model` | qwen2.5-7b-instruct | 模型名称 |
| `--num_samples` | 全部 | 处理样本数量 |
| `-m_turns` | 20 | 最大对话轮次 |
| `-o` | ../output-multiagentesc-aliyun | 输出目录 |

## 示例

```bash
# 测试单个样本
python inference-multiagentesc-aliyun.py --num_samples 1

# 使用更强的模型
python inference-multiagentesc-aliyun.py --model qwen2.5-32b-instruct

# 指定 API Key
python inference-multiagentesc-aliyun.py --api_key sk-your-key

# 完整配置
python inference-multiagentesc-aliyun.py \
    --model qwen2.5-14b-instruct \
    --num_samples 10 \
    -m_turns 15 \
    -o ../output-test
```

## 运行测评

```bash
cd ../evaluation

# CTRS 评估
python run_ctrs.py -i ../output-multiagentesc-aliyun -o ../output-ctrs-aliyun

# PANAS 评估
python PANAS/run_panas_before.py -i ../output-multiagentesc-aliyun -o ../output-panas-before-aliyun
python PANAS/run_panas_after.py -i ../output-multiagentesc-aliyun -o ../output-panas-after-aliyun

# WAI 评估
python WAI/run_wai.py -i ../output-multiagentesc-aliyun -o ../output-wai-aliyun

# Diversity 评估
python Diversity/run_diversity.py -i ../output-multiagentesc-aliyun -o ../output-diversity-aliyun
```

## 与完整版 MultiAgentESC 的区别

| 特性 | 简化版 | 完整版 |
|------|--------|--------|
| 依赖 | 仅 OpenAI API | AutoGen + 多智能体 |
| 提示词 | ✅ 使用 MultiAgentESC | ✅ 使用 MultiAgentESC |
| 策略选择 | 简化（零样本） | 复杂（情感+原因+意图） |
| 速度 | ⚡ 快 | 🐢 慢 |
| 效果 | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 常见问题

**Q: 提示词用的是哪个？**
A: 使用 MultiAgentESC 的 `zero_shot` 提示词

**Q: 为什么不用完整版 MultiAgentESC？**
A: 完整版依赖 AutoGen，配置复杂。简化版只需 API Key 即可运行。

**Q: 可以升级到完整版吗？**
A: 可以。安装 `pyautogen` 并配置 `OAI_CONFIG_LIST`，然后使用 `inference-multiagentesc.py`

## 文件说明

- `inference-multiagentesc-aliyun.py` - 简化版主脚本
- `run_multiagentesc_aliyun.bat` - 一键运行脚本

## 输出格式

生成的 `session_N.json` 文件：

```json
{
    "example": { ... },
    "cbt_technique": "MultiAgentESC-Aliyun (Strategy-based)",
    "cbt_plan": "MultiAgentESC using Aliyun qwen2.5-7b-instruct...",
    "cost": 0,
    "history": [
        {"role": "counselor", "message": "..."},
        {"role": "client", "message": "..."}
    ]
}
```
