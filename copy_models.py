"""
自动下载模型文件到 agent-tutorial 目录
"""
import os
import shutil

# 水溶解度模型文件
solubility_src = r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\水溶解度预测\模型数据'
solubility_dst = r'D:\00全部文件\agent-tutorial\solubility_models'

# 沸点模型文件
boiling_src = r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\沸点预测\模型数据'
boiling_dst = r'D:\00全部文件\agent-tutorial\boiling_models'

# 创建目标目录
os.makedirs(solubility_dst, exist_ok=True)
os.makedirs(boiling_dst, exist_ok=True)

# 复制水溶解度模型文件
print("复制水溶解度模型文件...")
for file in ['logs_best_model_v8_optimized.pkl', 'logs_feature_names_v8_optimized.txt', 'solubility_metrics_table_real.csv']:
    src_path = os.path.join(solubility_src, file)
    dst_path = os.path.join(solubility_dst, file)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"  复制 {file}")

# 复制沸点模型文件
print("\n复制沸点模型文件...")
for file in ['lstm_v2_models.pkl', 'clean_data.csv'] + [f'best_lstm_fold{i}.pth' for i in range(1, 6)] + [f'char2idx_fold{i}.pkl' for i in range(1, 6)] + [f'meta_fold{i}.pkl' for i in range(1, 6)]:
    src_path = os.path.join(boiling_src, file)
    dst_path = os.path.join(boiling_dst, file)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"  复制 {file}")

print("\n完成!")
