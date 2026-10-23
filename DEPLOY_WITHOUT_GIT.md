# 无需 Git 的部署指南 - Streamlit Cloud

## 方法：使用 ZIP 文件手动部署

### 步骤 1: 准备部署包

我已经为你创建了部署包：
- 文件名：`molecular_property_prediction_deploy.zip`
- 大小：约 155MB
- 位置：`D:\00全部文件\agent-tutorial\molecular_property_prediction_deploy.zip`

### 步骤 2: 解压部署包

1. 右键点击 `molecular_property_prediction_deploy.zip`
2. 选择 "全部解压..."
3. 解压到一个文件夹（例如：`D:\00全部文件\agent-tutorial\deploy`）

解压后的文件结构：
```
deploy/
├── web_app.py
├── requirements.txt
├── Procfile
├── .streamlit/
│   └── config.toml
├── solubility_models/
│   ├── logs_best_model_v8_optimized.pkl
│   ├── logs_feature_names_v8_optimized.txt
│   └── solubility_metrics_table_real.csv
└── boiling_models/
    ├── lstm_v2_models.pkl
    ├── clean_data.csv
    ├── best_lstm_fold{1-5}.pth
    ├── char2idx_fold{1-5}.pkl
    └── meta_fold{1-5}.pkl
```

### 步骤 3: 在 Streamlit Cloud 创建新应用

1. 访问 https://streamlit.io/cloud
2. 登录（使用 GitHub 账号）
3. 点击 "New app" 按钮
4. 选择 "Deploy from GitHub" 或 "Deploy from ZIP file"

#### 选项 A: 使用 ZIP 文件（推荐）

1. 点击 "Deploy from ZIP file"
2. 上传 `molecular_property_prediction_deploy.zip`
3. 填写应用信息：
   - **App file**: `web_app.py`
   - **Environment**: Python 3.11（推荐）
4. 点击 "Deploy!"

#### 选项 B: 使用 GitHub（需要创建临时仓库）

1. 访问 https://github.com/new
2. 创建一个新的公开仓库（例如：`molecular-property-prediction-temp`）
3. 不要初始化 README、gitignore 或 license
4. 点击 "Create repository"
5. 在仓库页面，点击 "Add file" → "Upload files"
6. 将解压后的所有文件拖放到上传区域
7. 点击 "Commit changes"
8. 返回 Streamlit Cloud，选择 "Deploy from GitHub"
9. 选择你创建的仓库
10. 填写应用信息：
    - **App file**: `web_app.py`
    - **Environment**: Python 3.11（推荐）
11. 点击 "Deploy!"

### 步骤 4: 等待部署完成

- 部署过程通常需要 5-10 分钟
- Streamlit Cloud 会显示部署进度
- 部署完成后，你会看到一个公开的 URL

### 步骤 5: 访问你的应用

- 复制公开的 URL
- 发送给任何人，他们都可以访问和使用这个应用
- 应用会 7x24 小时运行，不需要你的电脑保持开启

## 更新应用

### 方法 1: 使用 ZIP 文件

1. 修改 `web_app.py` 文件
2. 重新创建 ZIP 包（使用 `create_deploy_package.py`）
3. 在 Streamlit Cloud 应用页面，点击 "Edit app"
4. 上传新的 ZIP 文件
5. 点击 "Save and redeploy"

### 方法 2: 使用 GitHub

1. 修改代码
2. 推送到 GitHub
3. Streamlit Cloud 会自动重新部署

## 注意事项

### 1. 模型文件大小

- 模型文件约 155MB，上传可能需要一些时间
- 确保你的网络连接稳定

### 2. 首次加载时间

- 首次访问应用时，需要加载模型，可能需要 10-30 秒
- 之后的请求会更快

### 3. 免费 tier 限制

- Streamlit Cloud 免费版有以下限制：
  - 应用空闲 5 分钟后会休眠
  - 每月有运行时间限制
  - 公开应用可以被任何人访问

## 故障排除

### 问题 1: 上传 ZIP 文件失败

**错误信息**：`File too large`

**解决方案**：
- Streamlit Cloud 对 ZIP 文件有大小限制
- 如果文件太大，可以尝试使用 GitHub 方式部署

### 问题 2: 应用无法加载模型

**错误信息**：`No such file or directory`

**解决方案**：
- 确保所有模型文件都已上传
- 检查文件路径是否正确

### 问题 3: 应用空闲后需要重新加载

**解决方案**：这是 Streamlit Cloud 的正常行为，点击刷新即可

## 联系支持

如果遇到问题，可以：
1. 查看 Streamlit Cloud 文档：https://docs.streamlit.io/streamlit-community-cloud
2. 访问 Streamlit 论坛：https://discuss.streamlit.io/
3. 查看错误日志获取更多信息

---

**恭喜！你已经完成了所有准备工作，可以开始部署了！**
