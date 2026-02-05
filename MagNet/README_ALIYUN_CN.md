# MultiAgentESC + 阿里云 API 快速开始

## 一分钟配置

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

完成！对话结果保存在 `output-multiagentesc-aliyun/` 目录。

## 手动运行（更多控制）

```bash
cd MagNet/src

# 基本用法
python inference-multiagentesc-aliyun.py

# 自定义参数
python inference-multiagentesc-aliyun.py \
    --model qwen2.5-32b-instruct \
    --num_samples 10 \
    -m_turns 15
```

## 运行测评

```bash
cd ../evaluation

# CTRS 评估
python run_ctrs.py -i ../output-multiagentesc-aliyun -o ../output-ctrs-aliyun

# 其他测评
python PANAS/run_panas_before.py -i ../output-multiagentesc-aliyun -o ../output-panas-before-aliyun
python WAI/run_wai.py -i ../output-multiagentesc-aliyun -o ../output-wai-aliyun
```

## 支持的模型

- `qwen2.5-7b-instruct` - 快速经济 ✅
- `qwen2.5-14b-instruct` - 平衡选择
- `qwen2.5-32b-instruct` - 更好效果
- `qwen2.5-72b-instruct` - 最佳效果
- `qwen-max` - 最强模型

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--api_key` | API Key | 环境变量 |
| `--model` | 模型名称 | qwen2.5-7b-instruct |
| `--num_samples` | 样本数量 | 全部 |
| `-m_turns` | 最大轮次 | 20 |
| `-num_pr` | 并行进程数 | 1 |
| `-o` | 输出目录 | ../output-multiagentesc-aliyun |

## 快速测试

```bash
# 只处理 1 个样本，测试配置是否正确
python inference-multiagentesc-aliyun.py --num_samples 1
```

## 说明

**本脚本特点**：
- ✅ 使用 MultiAgentESC 的提示词系统
- ✅ 简化版，无需 AutoGen
- ✅ 配置简单，只需 API Key
- ✅ 与 MagNet ClientAgent 对话

**与完整 MultiAgentESC 的区别**：
- 简化版只使用零样本提示词
- 完整版使用复杂的多智能体协作
- 但两者的提示词是相同的

## 文档

- [简化版说明](README_ALIYUN_SIMPLE.md)
- [详细配置指南](ALIYUN_SETUP.md)
- [脚本对比](SCRIPTS_COMPARISON.md)

## 常见问题

**Q: API Key 在哪里获取？**
A: 访问 https://dashscope.aliyun.com/ 获取

**Q: 提示词用的是哪个？**
A: MultiAgentESC 自己的提示词系统（零样本）

**Q: 如何查看结果？**
A: 打开 `output-multiagentesc-aliyun/session_*.json` 文件

**Q: 出错了怎么办？**
A: 查看 `error_multiagentesc_aliyun_*.txt` 文件
