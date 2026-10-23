# 分子性质预测 Web 应用

这是一个基于 Streamlit 的分子性质预测工具，包含**沸点预测**和**水溶解度预测**两个功能。

## 🚀 快速开始

### 本地运行

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **运行应用**
```bash
streamlit run web_app.py
```

3. **访问应用**
打开浏览器，访问: `http://localhost:8501`

## 📦 功能说明

### 1. 沸点预测
- **输入**: 化学式或 SMILES
- **输出**: 沸点 (K, °C, °F)
- **方法**: 集成学习 (ExtraTrees + GradientBoosting + RandomForest)

### 2. 水溶解度预测
- **输入**: SMILES
- **输出**: LogS (水溶解度)
- **方法**: XGBoost + RF + MLP + Stacking (Ridge)
- **性能**: R² = 0.8137 (测试集)

## 📝 输入格式

### 沸点预测
- **化学式**: `C2H6O`, `C6H6`, `H2O`
- **SMILES**: `CCO`, `c1ccccc1`, `O`

### 水溶解度预测
- **SMILES**: `CCO`, `c1ccccc1`, `O=C=O`

## 📊 模型性能

### 水溶解度预测
| 模型 | R² (测试集) | MAE | RMSE |
|------|-------------|-----|------|
| XGBoost | 0.8137 | 0.69 | 1.01 |
| RF | 0.7764 | 0.79 | 1.11 |
| MLP | 0.7448 | 0.81 | 1.19 |
| Ensemble | 0.6564 | 1.05 | 1.38 |

## 🌐 部署到互联网

### 方法 1: Hugging Face Spaces (推荐)

1. 注册 Hugging Face 账号: https://huggingface.co/
2. 安装 Hugging Face CLI:
```bash
pip install huggingface_hub
huggingface-cli login
```

3. 创建新空间:
```bash
huggingface-cli repo create molecular-property-predictor --type space --sdk streamlit
```

4. 推送代码:
```bash
git clone https://huggingface.co/spaces/你的用户名/molecular-property-predictor
cd molecular-property-predictor
cp ../web_app.py .
cp ../requirements.txt .
git add .
git commit -m "Initial commit"
git push
```

5. 访问: `https://你的用户名.huggingface.co/molecular-property-predictor`

### 方法 2: Streamlit Cloud

1. 将代码上传到 GitHub
2. 访问: https://streamlit.io/cloud
3. 创建新应用，选择你的仓库和 `web_app.py`
4. 添加依赖: `streamlit`, `rdkit`, `scikit-learn`, `xgboost`

### 方法 3: 本地服务器

```bash
# 安装 nginx (反向代理)
sudo apt install nginx

# 安装 gunicorn (WSGI 服务器)
pip install gunicorn

# 启动应用
nohup streamlit run web_app.py --server.port 8501 > app.log 2>&1 &
```

## 📁 项目结构

```
agent-tutorial/
├── web_app.py           # 主应用文件
├── requirements.txt     # 依赖文件
├── README.md           # 说明文档
└── 模型数据/
    ├── 沸点预测/
    │   └── 模型数据/
    │       ├── best_lstm_fold4.pth
    │       ├── char2idx_fold4.pkl
    │       └── meta_fold4.pkl
    └── 水溶解度预测/
        └── 模型数据/
            ├── logs_best_model_v8_optimized.pkl
            ├── logs_all_models_v8_optimized.pkl
            └── logs_feature_names_v8_optimized.txt
```

## 🔧 技术栈

- **Python 3.8+**
- **Streamlit** - Web UI
- **RDKit** - 分子处理
- **Scikit-learn** - 机器学习
- **XGBoost** - 梯度提升
- **PyTorch** - LSTM 沸点预测

## 📄 许可证

本项目仅用于教学和科研目的。

## 👨‍💻 作者

徐杰生 - 天津大学

---

**注意**: 本工具预测结果仅供参考，实际应用中请结合实验数据验证。
