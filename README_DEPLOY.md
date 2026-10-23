# 分子性质预测工具

一个基于 Streamlit 的 Web 应用，用于预测分子的水溶解度和沸点。

## 功能

- **水溶解度预测 (LogS)**: 使用 XGBoost + RF + MLP + Stacking 集成模型
- **沸点预测**: 使用 LSTM (5折集成) 模型
- **输入格式支持**: 中文名、英文名、化学式、SMILES、CAS No
- **化学式搜索**: 当输入化学式时，显示所有匹配的异构体供选择

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run web_app.py
```

应用将在 `http://localhost:8501` 上运行。

## 部署到 Streamlit Cloud

### 步骤 1: 创建 GitHub 仓库

1. 将此项目上传到 GitHub 仓库
2. 确保包含以下文件：
   - `web_app.py`
   - `requirements.txt`
   - `Procfile`
   - `.streamlit/config.toml`
   - `solubility_models/` 目录（包含水溶解度模型文件）
   - `boiling_models/` 目录（包含沸点模型文件）

### 步骤 2: 部署到 Streamlit Cloud

1. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
2. 点击 "New app" 
3. 选择你的 GitHub 仓库
4. 配置应用：
   - Repository: `your-username/your-repo`
   - Main file path: `web_app.py`
   - Environment variables: （可选）
   - Advanced settings: （根据需要配置）

### 步骤 3: 等待部署完成

部署完成后，Streamlit Cloud 将提供一个公开的 URL，你可以分享给任何人。

## 注意事项

- 模型文件较大（约 200MB），建议使用 Git LFS 来管理
- 首次加载模型可能需要一些时间
- 公开部署后，任何人都可以访问和使用这个应用

## 技术栈

- **Streamlit**: Web 应用框架
- **RDKit**: 分子处理和化学式生成
- **XGBoost + RF + MLP**: 水溶解度预测模型
- **LSTM**: 沸点预测模型
- **Scikit-learn**: 机器学习工具
- **PyTorch**: 深度学习框架

## 许可证

MIT License
