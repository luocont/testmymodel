# MagNet 中文版快速配置指南

## ✅ 配置已完成

您的 MagNet 中文版已经配置完成！

### 📋 当前配置

#### API 配置 (.env 文件)
```bash
# 多智能体框架（阿里云百炼）
LLM_PROVIDER=aliyun
LLM_API_KEY=sk-40fb3997d3ed485ba390a9c4ae3bd2d2
LLM_MODEL=qwen2.5-7b-instruct

# 评估系统（OpenRouter）
EVAL_LLM_PROVIDER=openrouter
EVAL_LLM_API_KEY=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
EVAL_LLM_MODEL=openai/gpt-4o
```

#### 中文版文件
- **数据文件**: [dataset/data_cn.json](dataset/data_cn.json) - 9 个中文案例
- **提示词目录**: [prompts/cn/](prompts/cn/) - 中文提示词
- **推理脚本**: [src/inference-parallel-magnet-cn.py](src/inference-parallel-magnet-cn.py) - 中文版专用脚本

---

## 🚀 快速开始

### Windows 用户

双击运行启动脚本：
```
run_cn.bat
```

### Linux/Mac 用户

```bash
chmod +x run_cn.sh
./run_cn.sh
```

### 手动运行

```bash
cd src
python inference-parallel-magnet-cn.py -o ../output-cn -num_pr 4 -m_turns 20
```

---

## 📂 文件结构

```
MagNet/
├── dataset/
│   ├── data.json              # 英文数据
│   └── data_cn.json           # 中文数据（新增）
├── prompts/
│   ├── agent_*.txt           # 英文提示词
│   └── cn/                   # 中文提示词目录（新增）
│       ├── agent_client.txt
│       ├── agent_cbt.txt
│       ├── agent_technique.txt
│       ├── agent_reflections.txt
│       ├── agent_questioning.txt
│       ├── agent_solutions.txt
│       ├── agent_normalization.txt
│       ├── agent_psychoed.txt
│       └── agent_dialogue_gen.txt
├── src/
│   ├── inference-parallel-magnet.py      # 英文版
│   └── inference-parallel-magnet-cn.py   # 中文版（新增）
├── .env                                   # API 配置
├── run_cn.bat                            # Windows 启动脚本
└── run_cn.sh                             # Linux/Mac 启动脚本
```

---

## ⚙️ 配置说明

### 中文版脚本的特殊配置

在 `inference-parallel-magnet-cn.py` 中：

```python
# 数据文件路径使用中文版
DATA_FILE = "../dataset/data_cn.json"

# 提示词目录使用中文版
PROMPTS_DIR = "../prompts/cn/"
```

### 参数说明

运行脚本时的参数：

- `-o` / `--output_dir`: 输出目录（默认：当前目录）
- `-num_pr` / `--num_processes`: 并行进程数（默认：CPU 核心数）
- `-m_turns` / `--max_turns`: 最大对话轮数（默认：20）

---

## 📊 生成结果

### 输出目录
```
output-cn/
├── session_1.json
├── session_2.json
├── ...
└── session_N.json
```

### 输出格式
每个 session 文件包含：
```json
{
    "example": "客户初始信息",
    "cbt_technique": "使用的 CBT 技术",
    "cbt_plan": "咨询计划",
    "cost": "成本（美元）",
    "history": [
        {"role": "counselor", "message": "咨询师回应"},
        {"role": "client", "message": "客户回应"},
        ...
    ]
}
```

---

## 🔧 常见问题

### Q1: 如何更换模型？

**A:** 编辑 `.env` 文件：
```bash
# 使用其他阿里云模型
LLM_MODEL=qwen-plus

# 或使用 qwen-max（更强）
LLM_MODEL=qwen-max
```

### Q2: 如何调整并行数？

**A:** 修改启动脚本中的 `-num_pr` 参数：
```bash
python inference-parallel-magnet-cn.py -o ../output-cn -num_pr 2 -m_turns 20
```

### Q3: 如何调整对话轮数？

**A:** 修改 `-m_turns` 参数：
```bash
python inference-parallel-magnet-cn.py -o ../output-cn -num_pr 4 -m_turns 30
```

### Q4: API 调用失败怎么办？

**A:** 检查：
1. API 密钥是否正确
2. 网络连接是否正常
3. API 额度是否充足

### Q5: 如何添加更多中文案例？

**A:** 按照 `data_cn.json` 的格式添加新案例到该文件中。

---

## 💡 提示

### 推荐模型配置

**经济型配置：**
```bash
LLM_MODEL=qwen2.5-7b-instruct  # 对话生成
```

**高性能配置：**
```bash
LLM_MODEL=qwen-max  # 更强的推理能力
```

### 成本估算

| 组件 | 模型 | 参考成本 |
|------|------|----------|
| 对话生成 | qwen2.5-7b-instruct | ~¥0.004/千 tokens |
| 技术选择 | qwen-max | ~¥0.02/千 tokens |
| 评估 | gpt-4o | ~$0.005/千 tokens |

---

## 📚 相关文档

- [API 配置指南](docs/API_CONFIG_GUIDE.md)
- [中文版详细指南](docs/CHINESE_VERSION_GUIDE.md)
- [快速开始指南](docs/QUICK_START_CN.md)
