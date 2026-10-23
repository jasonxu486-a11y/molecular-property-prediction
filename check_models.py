"""
检查模型文件是否都存在
"""
import os

# 检查水溶解度模型文件
solubility_model_dir = r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\水溶解度预测\模型数据'
print("水溶解度模型文件:")
for file in ['logs_best_model_v8_optimized.pkl', 'logs_feature_names_v8_optimized.txt']:
    path = os.path.join(solubility_model_dir, file)
    exists = os.path.exists(path)
    print(f"  {file}: {'✓' if exists else '✗'}")

# 检查沸点模型文件
boiling_model_dir = r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\沸点预测\模型数据'
print("\n沸点模型文件:")
for file in ['lstm_v2_models.pkl', 'clean_data.csv'] + [f'best_lstm_fold{i}.pth' for i in range(1, 6)]:
    path = os.path.join(boiling_model_dir, file)
    exists = os.path.exists(path)
    print(f"  {file}: {'✓' if exists else '✗'}")
