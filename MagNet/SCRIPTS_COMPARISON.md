# MagNet 对话生成脚本对比

本目录包含多种对话生成脚本，可以根据不同的后端和需求选择使用。

## 可用脚本

### 1. MultiAgentESC + 阿里云 API ⭐ 推荐

**脚本**: `inference-multiagentesc-aliyun.py`

**特点**:
- ✅ 使用阿里云 API（无需本地部署）
- ✅ MultiAgentESC 自己的提示词系统
- ✅ 支持多种 Qwen 模型
- ✅ 配置简单（只需 API Key）
- ✅ 适合云环境使用

**运行方式**:
```bash
# 设置环境变量
set DASHSCOPE_API_KEY=your-key

# 运行
python inference-multiagentesc-aliyun.py

# 或使用一键脚本
run_multiagentesc_aliyun.bat
```

**文档**: [README_ALIYUN_CN.md](README_ALIYUN_CN.md)

---

### 2. MultiAgentESC + Ollama 本地部署

**脚本**: `inference-multiagentesc.py`

**特点**:
- ✅ MultiAgentESC 自己的提示词系统
- ✅ 完全本地运行（无需联网）
- ✅ 无 API 调用费用
- ❌ 需要安装 Ollama
- ❌ 需要本地 GPU（建议）
- ❌ 需要 OAI_CONFIG_LIST 配置

**运行方式**:
```bash
# 1. 安装 Ollama
# 2. 拉取模型: ollama pull qwen2.5:32b
# 3. 配置 OAI_CONFIG_LIST
# 4. 运行
python inference-multiagentesc.py --llm_model "qwen2.5:32b"
```

**文档**: [MULTIAGENTESC_INTEGRATION.md](MULTIAGENTESC_INTEGRATION.md)

---

### 3. MagNet 原始方法 + 阿里云

**脚本**: `inference-parallel-magnet.py`

**特点**:
- ✅ MagNet 官方方法
- ✅ 支持多种 LLM 后端
- ✅ 完整的测评验证
- ❌ 使用 MagNet 的提示词（不是 MultiAgentESC）

**运行方式**:
```bash
# 设置环境变量支持阿里云
set LLM_PROVIDER=aliyun
set LLM_API_KEY=your-key

# 运行
python inference-parallel-magnet.py -o ../output-magnet
```

**文档**: [README.md](README.md)

---

### 4. MagNet 原始方法 + 本地 vLLM

**脚本**: `inference-parallel-magnet.py`

**特点**:
- ✅ MagNet 官方方法
- ✅ 本地部署
- ❌ 需要 vLLM 配置

**运行方式**:
```bash
# 使用默认本地配置
python inference-parallel-magnet.py
```

---

## 快速选择指南

### 我应该使用哪个脚本？

| 场景 | 推荐脚本 | 模型 |
|------|---------|------|
| **快速测试** | `inference-multiagentesc-aliyun.py` | qwen2.5-7b-instruct |
| **正式测评** | `inference-multiagentesc-aliyun.py` | qwen2.5-32b-instruct |
| **无网络环境** | `inference-multiagentesc.py` | qwen2.5:32b (Ollama) |
| **对比 MagNet** | `inference-parallel-magnet.py` | 根据配置 |
| **成本优先** | `inference-multiagentesc.py` | 本地模型 |

### 按需求选择

#### 需要 MultiAgentESC 的提示词系统？

✅ 使用：
- `inference-multiagentesc-aliyun.py`（阿里云 API）
- `inference-multiagentesc.py`（Ollama 本地）

❌ 不使用：
- `inference-parallel-magnet.py`（MagNet 自己的提示词）

#### 需要最简单的配置？

✅ 使用 `inference-multiagentesc-aliyun.py`:
```bash
# 只需两步
set DASHSCOPE_API_KEY=your-key
python inference-multiagentesc-aliyun.py
```

#### 需要完全本地运行？

✅ 使用 `inference-multiagentesc.py`:
- 需要 Ollama
- 需要 GPU（推荐）
- 配置 `OAI_CONFIG_LIST`

#### 需要对比不同方法？

✅ 运行多个脚本，使用相同测评：
```bash
# MultiAgentESC (阿里云)
python inference-multiagentesc-aliyun.py -o ../output-multiagentesc-aliyun

# MultiAgentESC (Ollama)
python inference-multiagentesc.py -o ../output-multiagentesc

# MagNet 原始
python inference-parallel-magnet.py -o ../output-magnet

# 使用相同测评
cd ../evaluation
python run_ctrs.py -i ../output-multiagentesc-aliyun -o ../output-ctrs-aliyun
python run_ctrs.py -i ../output-multiagentesc -o ../output-ctrs
python run_ctrs.py -i ../output-magnet -o ../output-ctrs-magnet
```

---

## 参数对比

### 通用参数

所有脚本都支持以下参数：

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `-o` | 输出目录 | `../output-xxx` |
| `-d` | 数据集文件 | `../dataset/data_cn.json` |
| `-m_turns` | 最大对话轮次 | `20` |
| `-num_pr` | 并行进程数 | `4` |

### 脚本特定参数

#### inference-multiagentesc-aliyun.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--api_key` | 阿里云 API Key | 环境变量 |
| `--model` | 模型名称 | qwen2.5-7b-instruct |
| `--num_samples` | 样本数量 | 全部 |

#### inference-multiagentesc.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--llm_model` | LLM 模型 | qwen2.5:32b |
| `--model_path` | 嵌入模型 | all-roberta-large-v1 |
| `--cache_path` | 缓存路径 | "" |

#### inference-parallel-magnet.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 依赖环境变量 | `LLM_PROVIDER` | local |
| 依赖环境变量 | `LLM_API_KEY` | - |
| 依赖环境变量 | `LLM_MODEL` | gpt-4o-mini |

---

## 输出格式

所有脚本生成的输出格式相同，都兼容 MagNet 的测评脚本：

```json
{
    "example": { ... },
    "cbt_technique": "技术名称",
    "cbt_plan": "计划描述",
    "cost": 0,
    "history": [
        {"role": "counselor", "message": "..."},
        {"role": "client", "message": "..."}
    ]
}
```

---

## 成本对比

| 脚本 | 成本类型 | 说明 |
|------|---------|------|
| `inference-multiagentesc-aliyun.py` | API 调用费 | 按 tokens 计费 |
| `inference-multiagentesc.py` | 硬件 + 电费 | 本地运行 |
| `inference-parallel-magnet.py` | 取决于配置 | API 或本地 |

### 阿里云 API 成本估算（qwen2.5-7b-instruct）

- 输入: ¥0.0005/千tokens
- 输出: ¥0.002/千tokens
- 每个对话: 约 ¥0.01-0.03
- 100 个对话: 约 ¥1-3

---

## 性能对比

| 脚本 | 速度 | 质量 | 稳定性 |
|------|------|------|--------|
| MultiAgentESC + 阿里云 7B | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| MultiAgentESC + 阿里云 32B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| MultiAgentESC + Ollama | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| MagNet 原始 + 阿里云 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 测评兼容性

所有脚本生成的输出都可以用相同的测评脚本：

```bash
cd evaluation

# CTRS
python run_ctrs.py -i ../output-XXX -o ../output-ctrs-XXX

# PANAS
python PANAS/run_panas_before.py -i ../output-XXX -o ../output-panas-before-XXX
python PANAS/run_panas_after.py -i ../output-XXX -o ../output-panas-after-XXX

# WAI
python WAI/run_wai.py -i ../output-XXX -o ../output-wai-XXX

# Diversity
python Diversity/run_diversity.py -i ../output-XXX -o ../output-diversity-XXX
```

---

## 迁移指南

### 从 MultiAgentESC + Ollama 迁移到阿里云

**之前** (Ollama):
```bash
python inference-multiagentesc.py --llm_model "qwen2.5:32b"
```

**之后** (阿里云):
```bash
set DASHSCOPE_API_KEY=your-key
python inference-multiagentesc-aliyun.py --model qwen2.5-32b-instruct
```

### 从 MagNet 原始迁移到 MultiAgentESC

**之前** (MagNet):
```bash
set LLM_PROVIDER=aliyun
set LLM_API_KEY=your-key
python inference-parallel-magnet.py
```

**之后** (MultiAgentESC):
```bash
set DASHSCOPE_API_KEY=your-key
python inference-multiagentesc-aliyun.py
```

---

## 故障排除

### 脚本选择问题

**Q**: 我不知道该用哪个脚本？
**A**: 从 `inference-multiagentesc-aliyun.py` 开始，配置最简单。

**Q**: 需要最便宜的方案？
**A**: 使用 `inference-multiagentesc.py` + 本地 Ollama。

**Q**: 需要最好的效果？
**A**: 使用 `inference-multiagentesc-aliyun.py --model qwen-max`。

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `API Key not found` | 未设置 API Key | 设置环境变量或使用 `--api_key` |
| `Model not found` | 模型名称错误 | 检查模型名称拼写 |
| `Connection error` | 网络问题 | 检查网络连接 |
| `Out of memory` | 本地模型太大 | 换用更小的模型 |

---

## 最佳实践

1. **测试阶段**: 使用 `--num_samples 1` 快速验证
2. **开发阶段**: 使用 `qwen2.5-7b-instruct` 节省成本
3. **生产阶段**: 使用 `qwen2.5-32b-instruct` 平衡效果和成本
4. **对比研究**: 运行多个脚本，使用相同测评

---

## 相关文档

- [阿里云配置指南](ALIYUN_SETUP.md)
- [MultiAgentESC 集成说明](MULTIAGENTESC_INTEGRATION.md)
- [详细使用指南](MULTIAGENTESC_USAGE.md)
- [MagNet 原始文档](README.md)
