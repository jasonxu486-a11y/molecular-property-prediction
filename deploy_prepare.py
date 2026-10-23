#!/usr/bin/env python
"""
部署准备脚本 - 检查所有必要文件
"""
import os
import sys

def check_file(path, description):
    """检查文件是否存在"""
    if os.path.exists(path):
        print(f"[✓] {description}")
        return True
    else:
        print(f"[✗] {description} - 不存在")
        return False

def main():
    print("=" * 60)
    print("分子性质预测工具 - 部署准备")
    print("=" * 60)
    print()
    
    # 检查必要文件
    print("检查必要文件...")
    checks = [
        ("web_app.py", "主应用文件"),
        ("requirements.txt", "依赖文件"),
        ("Procfile", "Streamlit Cloud 配置"),
        (".streamlit/config.toml", "Streamlit 配置"),
        ("solubility_models", "水溶解度模型目录"),
        ("boiling_models", "沸点模型目录"),
    ]
    
    all_ok = True
    for path, desc in checks:
        if not check_file(path, desc):
            all_ok = False
    
    print()
    
    # 检查模型文件
    print("检查模型文件...")
    solubility_files = [
        "logs_best_model_v8_optimized.pkl",
        "logs_feature_names_v8_optimized.txt",
    ]
    
    for file in solubility_files:
        path = os.path.join("solubility_models", file)
        if not check_file(path, f"水溶解度模型: {file}"):
            all_ok = False
    
    boiling_files = [
        "lstm_v2_models.pkl",
        "clean_data.csv",
    ] + [f"best_lstm_fold{i}.pth" for i in range(1, 6)] + \
        [f"char2idx_fold{i}.pkl" for i in range(1, 6)] + \
        [f"meta_fold{i}.pkl" for i in range(1, 6)]
    
    for file in boiling_files:
        path = os.path.join("boiling_models", file)
        if not check_file(path, f"沸点模型: {file}"):
            all_ok = False
    
    print()
    
    # 检查 Python 环境
    print("检查 Python 环境...")
    try:
        import subprocess
        result = subprocess.run(["python", "--version"], capture_output=True, text=True)
        print(f"[✓] {result.stdout.strip()}")
    except Exception as e:
        print(f"[✗] Python 未安装: {e}")
        all_ok = False
    
    # 检查依赖
    try:
        import streamlit
        print(f"[✓] streamlit 已安装 (v{streamlit.__version__})")
    except ImportError:
        print("[⚠] streamlit 未安装，运行 pip install -r requirements.txt 安装依赖")
    
    try:
        import rdkit
        print(f"[✓] rdkit 已安装 (v{rdkit.__version__})")
    except ImportError:
        print("[⚠] rdkit 未安装，运行 pip install -r requirements.txt 安装依赖")
    
    try:
        import torch
        print(f"[✓] torch 已安装 (v{torch.__version__})")
    except ImportError:
        print("[⚠] torch 未安装，运行 pip install -r requirements.txt 安装依赖")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("部署准备完成！")
        print()
        print("下一步：")
        print("1. 初始化 Git 仓库：git init")
        print("2. 添加文件：git add .")
        print("3. 提交更改：git commit -m 'Initial commit'")
        print("4. 创建 GitHub 仓库并推送代码")
        print("5. 部署到 Streamlit Cloud：https://streamlit.io/cloud")
        print()
        print("详细步骤请参考 DEPLOY_GUIDE.md")
    else:
        print("部署准备失败，请检查上述错误")
    
    print("=" * 60)
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
