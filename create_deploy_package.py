#!/usr/bin/env python
"""
创建部署包 - 将所有必要文件打包成 ZIP
"""
import os
import zipfile
from pathlib import Path

def create_deploy_package():
    """创建部署包"""
    package_name = "molecular_property_prediction_deploy.zip"
    
    # 必要的文件和目录
    files_to_include = [
        "web_app.py",
        "requirements.txt",
        "Procfile",
        (".streamlit", ".streamlit/config.toml"),
    ]
    
    directories_to_include = [
        "solubility_models",
        "boiling_models",
    ]
    
    # 创建 ZIP 文件
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加文件
        for item in files_to_include:
            if isinstance(item, tuple):
                # 相对路径
                src = item[1]
                arcname = item[1]
            else:
                src = item
                arcname = item
            
            if os.path.exists(src):
                zipf.write(src, arcname)
                print(f"添加: {src}")
            else:
                print(f"警告: {src} 不存在")
        
        # 添加目录
        for dir_name in directories_to_include:
            if os.path.exists(dir_name):
                for root, dirs, files in os.walk(dir_name):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, ".")
                        zipf.write(file_path, arcname)
                        print(f"添加: {file_path}")
            else:
                print(f"警告: {dir_name} 不存在")
    
    print(f"\n部署包已创建: {package_name}")
    print("大小: {:.2f} MB".format(os.path.getsize(package_name) / (1024 * 1024)))
    
    return package_name

if __name__ == "__main__":
    create_deploy_package()
