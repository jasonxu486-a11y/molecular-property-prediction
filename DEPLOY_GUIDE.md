# 部署到 Streamlit Cloud 完整指南

## 准备工作

1. **确保所有文件都在 `agent-tutorial` 目录下**：
   - `web_app.py` ✓
   - `requirements.txt` ✓
   - `Procfile` ✓
   - `.streamlit/config.toml` ✓
   - `solubility_models/` ✓
   - `boiling_models/` ✓

2. **安装 Git**（如果尚未安装）

## 部署步骤

### 步骤 1: 初始化 Git 仓库

```bash
cd D:\00全部文件\agent-tutorial
git init
git add .
git commit -m "Initial commit: Molecular property prediction app"
```

### 步骤 2: 创建 GitHub 仓库

1. 访问 https://github.com/new
2. 创建一个新的公开仓库（例如：`molecular-property-prediction`）
3. 不要初始化 README、gitignore 或 license
4. 复制仓库 URL（例如：`https://github.com/your-username/molecular-property-prediction.git`）

### 步骤 3: 推送代码到 GitHub

```bash
git remote add origin https://github.com/your-username/molecular-property-prediction.git
git branch -M main
git push -u origin main
```

如果遇到大文件问题（模型文件约 200MB），需要安装 Git LFS：

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pkl"
git lfs track "*.pth"

# 重新添加文件
git add .
git commit -m "Add models with Git LFS"
git push -u origin main
```

### 步骤 4: 部署到 Streamlit Cloud

1. 访问 https://streamlit.io/cloud
2. 登录（使用 GitHub 账号）
3. 点击 "New app" 按钮
4. 填写应用信息：
   - **GitHub repository**: 选择你创建的仓库
   - **Main file path**: `web_app.py`
   - **Environment variables**: （留空）
   - **Advanced settings**:
     - Python version: `3.11`（推荐）或 `3.10`
     - Requirements file: `requirements.txt`

5. 点击 "Deploy!" 按钮

### 步骤 5: 等待部署完成

- 部署过程通常需要 5-10 分钟
- Streamlit Cloud 会显示部署进度
- 部署完成后，你会看到一个公开的 URL（例如：`https://your-username-molecular-property-prediction-app-streamlit-app-xxx.streamlit.app`）

### 步骤 6: 分享你的应用

- 复制公开的 URL
- 发送给任何人，他们都可以访问和使用这个应用
- 应用会 7x24 小时运行，不需要你的电脑保持开启

## 注意事项

### 1. 模型文件大小

- 模型文件约 200MB，可能超过 GitHub 的文件大小限制（100MB）
- 解决方案：使用 Git LFS（Large File Storage）

### 2. 首次加载时间

- 首次访问应用时，需要加载模型，可能需要 10-30 秒
- 之后的请求会更快

### 3. 免费 tier 限制

- Streamlit Cloud 免费版有以下限制：
  - 应用空闲 5 分钟后会休眠
  - 每月有运行时间限制（具体查看官网）
  - 公开应用可以被任何人访问

### 4. 更新应用

- 修改代码后，只需推送到 GitHub：
  ```bash
  git add .
  git commit -m "Update description"
  git push
  ```
- Streamlit Cloud 会自动重新部署

## 故障排除

### 问题 1: 模型加载失败

**错误信息**：`No such file or directory`

**解决方案**：
- 确保模型文件已推送到 GitHub
- 检查 `web_app.py` 中的路径是否正确

### 问题 2: 依赖安装失败

**错误信息**：`Failed to install packages`

**解决方案**：
- 检查 `requirements.txt` 中的包名是否正确
- 确保包版本兼容

### 问题 3: 应用无法访问

**错误信息**：`Application error`

**解决方案**：
- 检查 Streamlit Cloud 的日志（点击 "View logs"）
- 查看错误信息并修复代码

## 联系支持

如果遇到问题，可以：
1. 查看 Streamlit Cloud 文档：https://docs.streamlit.io/streamlit-community-cloud
2. 访问 Streamlit 论坛：https://discuss.streamlit.io/
3. 查看错误日志获取更多信息
