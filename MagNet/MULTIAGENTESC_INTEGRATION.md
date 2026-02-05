# MultiAgentESC 集成说明

这个目录包含将 MultiAgentESC 框架集成到 MagNet 测评系统的脚本。

## 概述

这个集成方案让 MultiAgentESC 作为咨询师，与 MagNet 的 ClientAgent 进行对话，生成符合 MagNet 测评格式的对话文件。

**重要特点：**
- MultiAgentESC 使用自己的提示词系统（不使用 MagNet 的预设提示词）
- 输出格式完全兼容 MagNet 的测评脚本
- 可以直接使用 MagNet 的测评工具进行评估

## 框架对比

### MagNet
- 使用预设的 Agent 提示词（agent_client.txt, agent_cbt.txt 等）
- 基于 CBT 技术的多智能体系统
- 输出包含 example、cbt_technique、cbt_plan、cost、history

### MultiAgentESC
- 使用自己的提示词系统（prompt.py）
- 基于策略选择的多智能体协作系统
- 动态分析情感、原因、意图，选择合适的咨询策略

## 使用步骤

### 1. 环境准备

确保两个框架的依赖都已安装：

```bash
# MagNet 依赖
pip install -r requirements.txt

# MultiAgentESC 依赖
pip install autogen sentence-transformers
```

### 2. 配置 LLM

在 `MultiAgentESC` 目录下创建 `OAI_CONFIG_LIST` 文件：

```json
[
    {
        "model": "qwen2.5:32b",
        "base_url": "http://localhost:11434/v1",
        "api_type": "openai"
    }
]
```

### 3. 准备嵌入数据（可选）

MultiAgentESC 使用 sentence embeddings 进行策略检索。如果已有 `embeddings.txt`，请确保它在 `MultiAgentESC` 目录下。

如果没有，MultiAgentESC 会使用零样本生成作为备选方案。

### 4. 生成对话

```bash
cd MagNet/src

# 基本用法
python inference-multiagentesc.py

# 指定输出目录
python inference-multiagentesc.py -o ../output-multiagentesc

# 指定 LLM 模型
python inference-multiagentesc.py --llm_model "qwen2.5:32b"

# 指定最大对话轮次
python inference-multiagentesc.py -m_turns 10

# 多进程并行
python inference-multiagentesc.py -num_pr 4
```

### 5. 运行测评

对话生成完成后，使用 MagNet 的测评脚本：

```bash
cd ../evaluation

# CTRS 评估
python run_ctrs.py -i ../output-multiagentesc -o ../output-ctrs-multiagentesc

# PANAS 评估
python PANAS/run_panas_before.py -i ../output-multiagentesc -o ../output-panas-before-multiagentesc
python PANAS/run_panas_after.py -i ../output-multiagentesc -o ../output-panas-after-multiagentesc

# WAI 评估
python WAI/run_wai.py -i ../output-multiagentesc -o ../output-wai-multiagentesc

# Diversity 评估
python Diversity/run_diversity.py -i ../output-multiagentesc -o ../output-diversity-multiagentesc
```

## 输出格式

生成的 `session_N.json` 文件格式：

```json
{
    "example": {
        "AI_client": { ... },
        "AI_counselor": { ... },
        "ground_truth_CBT": [ ... ]
    },
    "cbt_technique": "MultiAgentESC (Strategy-based)",
    "cbt_plan": "MultiAgentESC uses dynamic strategy selection based on emotion, cause, and intention analysis.",
    "cost": 0,
    "history": [
        {"role": "counselor", "message": "..."},
        {"role": "client", "message": "..."},
        ...
    ]
}
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output_dir` | output-multiagentesc | 输出目录 |
| `-d, --dataset` | ../dataset/data_cn.json | 数据集文件 |
| `-num_pr, --num_processes` | None | 并行进程数 |
| `-m_turns, --max_turns` | 20 | 最大对话轮次 |
| `--llm_model` | qwen2.5:32b | LLM 模型名称 |
| `--model_path` | all-roberta-large-v1 | SentenceTransformer 模型 |
| `--cache_path` | "" | 缓存路径 |

## MultiAgentESC 工作流程

1. **复杂度判断**: 检查对话是否需要多智能体协作
2. **情感分析**: 识别用户的情感状态
3. **原因分析**: 分析导致情感的具体事件
4. **意图分析**: 推断用户的咨询意图
5. **策略选择**: 根据分析结果选择合适的咨询策略
6. **响应生成**: 使用选定的策略生成响应
7. **多策略辩论** (如需要): 如果多个策略适用，进行辩论和投票
8. **自我反思**: 对生成的响应进行反思和优化

## 常见问题

### Q: 提示词冲突？
A: 不会。MultiAgentESC 使用自己的提示词系统（在 `MultiAgentESC/prompt.py` 中定义），完全独立于 MagNet 的提示词。

### Q: 如何更换 LLM？
A: 修改 `OAI_CONFIG_LIST` 文件或使用 `--llm_model` 参数指定。

### Q: 速度太慢？
A: 使用 `-num_pr` 参数启用多进程并行，或者减少 `-m_turns` 限制对话轮次。

### Q: 出现错误？
A: 检查 `error_multiagentesc_*.txt` 文件查看详细错误信息。

## 与原始 MultiAgentESC 的区别

1. **接口适配**: 将 MultiAgentESC 适配为 MagNet 的咨询师接口
2. **格式兼容**: 输出格式符合 MagNet 测评要求
3. **保留核心**: 完整保留 MultiAgentESC 的提示词和策略系统
4. **简化成本**: 不计算 API 成本（cost=0）

## 引用

如果使用此集成方案，请同时引用两个框架：

```bibtex
@software{magnet2024,
  title = {MagNet: A Multi-Agent Framework for Mental Health Counseling},
  author = {...},
  year = {2024}
}

@software{multiagentesc2024,
  title = {Multi-Agent ESC: Emotional Support Strategy Selection},
  author = {...},
  year = {2024}
}
```
