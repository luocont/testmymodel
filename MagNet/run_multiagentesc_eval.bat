@echo off
REM ============================================
REM MultiAgentESC 测评一键运行脚本
REM ============================================

echo ========================================
echo MultiAgentESC 测评系统
echo ========================================
echo.

REM 设置 Python 路径（如果需要）
REM set PYTHON_PATH=python

REM 检查 Python 是否可用
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python
    pause
    exit /b 1
)

REM 1. 生成 embeddings（如果还没有）
echo [步骤 1/3] 检查 embeddings.txt...
if not exist "..\MultiAgentESC\embeddings.txt" (
    echo 未找到 embeddings.txt，正在生成...
    cd ..\MultiAgentESC
    python generate_embeddings.py --dataset dataset/ESConv.json --output embeddings.txt
    cd ..\MagNet\src
    if %ERRORLEVEL% neq 0 (
        echo 警告: embeddings.txt 生成失败，将使用零样本生成
    )
) else (
    echo embeddings.txt 已存在
)

REM 2. 生成对话
echo.
echo [步骤 2/3] 生成对话...
echo 使用 MultiAgentESC 作为咨询师...

REM 检查 OAI_CONFIG_LIST 是否存在
if not exist "..\MultiAgentESC\OAI_CONFIG_LIST" (
    echo.
    echo 警告: 未找到 OAI_CONFIG_LIST 文件！
    echo 请先在 MultiAgentESC 目录下创建 OAI_CONFIG_LIST 文件
    echo.
    echo 示例内容:
    echo [
    echo     {
    echo         "model": "qwen2.5:32b",
    echo         "base_url": "http://localhost:11434/v1",
    echo         "api_type": "openai"
    echo     }
    echo ]
    echo.
    pause
    exit /b 1
)

python inference-multiagentesc.py -o ../output-multiagentesc -m_turns 20

if %ERRORLEVEL% neq 0 (
    echo 错误: 对话生成失败
    pause
    exit /b 1
)

REM 3. 运行测评
echo.
echo [步骤 3/3] 运行测评...
echo.

REM 检查是否要运行所有测评
set /p RUN_ALL="是否运行所有测评？(Y/N): "

if /i "%RUN_ALL%"=="Y" (
    echo 运行 CTRS 评估...
    cd ..\evaluation
    python run_ctrs.py -i ../output-multiagentesc -o ../output-ctrs-multiagentesc -m_iter 1

    echo 运行 PANAS 评估...
    python PANAS/run_panas_before.py -i ../output-multiagentesc -o ../output-panas-before-multiagentesc
    python PANAS/run_panas_after.py -i ../output-multiagentesc -o ../output-panas-after-multiagentesc

    echo 运行 WAI 评估...
    python WAI/run_wai.py -i ../output-multiagentesc -o ../output-wai-multiagentesc

    echo 运行 Diversity 评估...
    python Diversity/run_diversity.py -i ../output-multiagentesc -o ../output-diversity-multiagentesc

    cd ..\src
) else (
    echo 跳过自动测评
    echo 你可以手动运行测评脚本:
    echo   cd ..\evaluation
    echo   python run_ctrs.py -i ../output-multiagentesc -o ../output-ctrs-multiagentesc
)

echo.
echo ========================================
echo 完成！
echo ========================================
echo 对话结果: ..\output-multiagentesc\
echo 测评结果: ..\output-ctrs-multiagentesc\ (等)
echo.
pause
