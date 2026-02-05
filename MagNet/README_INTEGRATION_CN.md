# MultiAgentESC 集成到 MagNet 测评系统

这是一个将 **MultiAgentESC** 框架集成到 **MagNet** 测评系统的完整方案。

## 核心特点

✅ **保留 MultiAgentESC 的提示词系统** - 完全使用 MultiAgentESC 自己的提示词
✅ **输出格式兼容 MagNet** - 生成的对话文件可以直接用 MagNet 的测评脚本评估
✅ **一键运行** - 提供自动化脚本，简化操作流程

## 快速开始

### 1. 配置 LLM

在 `MultiAgentESC/` 目录下创建 `OAI_CONFIG_LIST` 文件：

```json
[
    {
        "model": "qwen2.5:32b",
        "base_url": "http://localhost:11434/v1",
        "api_type": "openai"
    }
]
```

### 2. 一键运行

```bash
cd MagNet
run_multiagentesc_eval.bat
```

### 3. 查看结果

- 对话输出: `output-multiagentesc/`
- 测评结果: `output-ctrs-multiagentesc/` 等

## 手动运行

### 生成对话

```bash
cd MagNet/src
python inference-multiagentesc.py -o ../output-multiagentesc
```

### 运行测评

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

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/inference-multiagentesc.py` | 主集成脚本，让 MultiAgentESC 与 MagNet Client 对话 |
| `run_multiagentesc_eval.bat` | 一键运行脚本 |
| `MULTIAGENTESC_INTEGRATION.md` | 集成技术说明 |
| `MULTIAGENTESC_USAGE.md` | 详细使用指南 |
| `MultiAgentESC/generate_embeddings.py` | 生成策略检索所需的嵌入文件 |

## 工作流程

```
1. 配置 LLM (OAI_CONFIG_LIST)
   ↓
2. 生成 embeddings (可选)
   ↓
3. MultiAgentESC (咨询师) ↔ MagNet Client (客户)
   ↓
4. 输出 session_N.json
   ↓
5. 使用 MagNet 测评脚本评估
```

## MultiAgentESC 框架

MultiAgentESC 是一个基于策略选择的多智能体情感支持系统：

- **情感感知**: 识别用户情感状态
- **原因分析**: 分析导致情感的事件
- **意图推断**: 理解用户咨询意图
- **策略选择**: 选择合适的咨询策略（7种）
- **响应生成**: 通过辩论、投票、反思生成响应

**支持的策略**：
- Question（提问）
- Restatement or Paraphrasing（重述）
- Reflection of feelings（情感反映）
- Self-disclosure（自我披露）
- Affirmation and Reassurance（肯定安慰）
- Providing Suggestions（提供建议）
- Information（提供信息）

## 提示词系统

⚠️ **重要**: MultiAgentESC 使用自己的提示词系统（定义在 `MultiAgentESC/prompt.py`），完全不使用 MagNet 的预设提示词。

MultiAgentESC 的提示词包括：
- `behavior_control`: 判断对话复杂度
- `zero_shot`: 零样本生成
- `get_emotion`: 情感分析
- `get_cause`: 原因分析
- `get_intention`: 意图分析

## 参数说明

```bash
python inference-multiagentesc.py \
    -o ../output-multiagentesc \      # 输出目录
    -d ../dataset/data_cn.json \      # 数据集文件
    -m_turns 20 \                      # 最大对话轮次
    --llm_model "qwen2.5:32b" \       # LLM 模型
    -num_pr 4                          # 并行进程数
```

## 输出格式

生成的 `session_N.json` 文件格式：

```json
{
    "example": { ... },
    "cbt_technique": "MultiAgentESC (Strategy-based)",
    "cbt_plan": "MultiAgentESC uses dynamic strategy selection...",
    "cost": 0,
    "history": [
        {"role": "counselor", "message": "..."},
        {"role": "client", "message": "..."}
    ]
}
```

## 常见问题

**Q: 提示词会冲突吗？**
A: 不会。MultiAgentESC 完全使用自己的提示词系统。

**Q: embeddings.txt 是必须的吗？**
A: 不是必需的，但没有的话会降级到零样本生成，效果可能较差。

**Q: 如何加速处理？**
A: 使用 `-num_pr` 参数启用多进程，或减少 `-m_turns` 限制对话轮次。

**Q: 出现错误怎么办？**
A: 检查 `OAI_CONFIG_LIST` 文件，查看 `error_multiagentesc_*.txt` 了解详细错误。

## 对比测评

可以同时运行多个方法进行对比：

```bash
# MultiAgentESC
python inference-multiagentesc.py -o ../output-multiagentesc

# MagNet 原始方法
python inference-parallel-magnet.py -o ../output-magnet

# 使用相同测评
cd ../evaluation
python run_ctrs.py -i ../output-multiagentesc -o ../output-ctrs-multiagentesc
python run_ctrs.py -i ../output-magnet -o ../output-ctrs-magnet
```

## 更多文档

- [集成技术说明](MULTIAGENTESC_INTEGRATION.md)
- [详细使用指南](MULTIAGENTESC_USAGE.md)
- [MagNet 文档](README.md)

## 许可证

请同时遵守 MagNet 和 MultiAgentESC 的许可证要求。
