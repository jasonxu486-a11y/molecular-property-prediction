@echo off
echo ========================================
echo 分子性质预测工具 - 部署准备
echo ========================================
echo.

echo 检查必要文件...
if not exist "web_app.py" (
    echo [错误] web_app.py 不存在
    pause
    exit /b 1
)
if not exist "requirements.txt" (
    echo [错误] requirements.txt 不存在
    pause
    exit /b 1
)
if not exist "Procfile" (
    echo [错误] Procfile 不存在
    pause
    exit /b 1
)
if not exist ".streamlit\config.toml" (
    echo [错误] .streamlit/config.toml 不存在
    pause
    exit /b 1
)
if not exist "solubility_models" (
    echo [错误] solubility_models 目录不存在
    pause
    exit /b 1
)
if not exist "boiling_models" (
    echo [错误] boiling_models 目录不存在
    pause
    exit /b 1
)

echo [✓] 所有必要文件都存在
echo.

echo 检查模型文件...
if not exist "solubility_models\logs_best_model_v8_optimized.pkl" (
    echo [警告] solubility_models/logs_best_model_v8_optimized.pkl 不存在
)
if not exist "boiling_models\lstm_v2_models.pkl" (
    echo [警告] boiling_models/lstm_v2_models.pkl 不存在
)
echo [✓] 模型文件检查完成
echo.

echo 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python 未安装
    pause
    exit /b 1
)
echo [✓] Python 已安装

pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [警告] streamlit 未安装，运行 pip install -r requirements.txt 安装依赖
) else (
    echo [✓] streamlit 已安装
)
echo.

echo ========================================
echo 部署准备完成！
echo ========================================
echo.
echo 下一步：
echo 1. 初始化 Git 仓库：git init
echo 2. 添加文件：git add .
echo 3. 提交更改：git commit -m "Initial commit"
echo 4. 创建 GitHub 仓库并推送代码
echo 5. 部署到 Streamlit Cloud：https://streamlit.io/cloud
echo.
echo 详细步骤请参考 DEPLOY_GUIDE.md
echo.
pause
