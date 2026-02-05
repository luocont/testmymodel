@echo off
REM ============================================
REM MultiAgentESC + 阿里云 API 一键运行脚本
REM ============================================

echo ========================================
echo MultiAgentESC + 阿里云 API 测评系统
echo ========================================
echo.

REM 检查 API Key
if "%DASHSCOPE_API_KEY%"=="" (
    echo 错误: 未设置 DASHSCOPE_API_KEY 环境变量！
    echo.
    echo 请先设置阿里云 API Key:
    echo   set DASHSCOPE_API_KEY=your-api-key-here
    echo.
    echo 或者使用完整命令:
    echo   python inference-multiagentesc-aliyun.py --api_key your-api-key
    echo.
    pause
    exit /b 1
)

echo [信息] 使用阿里云 API Key: %DASHSCOPE_API_KEY:~0,10%...
echo.

REM 设置参数
set OUTPUT_DIR=..\output-multiagentesc-aliyun
set MODEL=qwen2.5-7b-instruct
set MAX_TURNS=20
set NUM_SAMPLES=3

echo 配置:
echo   输出目录: %OUTPUT_DIR%
echo   模型: %MODEL%
echo   最大轮次: %MAX_TURNS%
echo   样本数: %NUM_SAMPLES%
echo.

REM 生成对话
echo [步骤 1/2] 生成对话...
cd src
python inference-multiagentesc-aliyun.py ^
    -o %OUTPUT_DIR% ^
    --model %MODEL% ^
    -m_turns %MAX_TURNS% ^
    --num_samples %NUM_SAMPLES%

if %ERRORLEVEL% neq 0 (
    echo 错误: 对话生成失败
    pause
    exit /b 1
)

REM 运行测评
echo.
echo [步骤 2/2] 运行测评...
set /p RUN_EVAL="是否运行测评？(Y/N): "

if /i "%RUN_EVAL%"=="Y" (
    cd ..\evaluation

    echo 运行 CTRS 评估...
    python run_ctrs.py -i %OUTPUT_DIR% -o ..\output-ctrs-multiagentesc-aliyun -m_iter 1

    echo 运行 PANAS 评估...
    python PANAS\run_panas_before.py -i %OUTPUT_DIR% -o ..\output-panas-before-multiagentesc-aliyun
    python PANAS\run_panas_after.py -i %OUTPUT_DIR% -o ..\output-panas-after-multiagentesc-aliyun

    echo 运行 WAI 评估...
    python WAI\run_wai.py -i %OUTPUT_DIR% -o ..\output-wai-multiagentesc-aliyun

    echo 运行 Diversity 评估...
    python Diversity\run_diversity.py -i %OUTPUT_DIR% -o ..\output-diversity-multiagentesc-aliyun

    cd ..\src
) else (
    echo 跳过测评
    echo 你可以手动运行测评:
    echo   cd ..\evaluation
    echo   python run_ctrs.py -i %OUTPUT_DIR% -o ..\output-ctrs-multiagentesc-aliyun
)

echo.
echo ========================================
echo 完成！
echo ========================================
echo 对话结果: %OUTPUT_DIR%\
if /i "%RUN_EVAL%"=="Y" (
    echo 测评结果: ..\output-ctrs-multiagentesc-aliyun\ (等)
)
echo.
pause
