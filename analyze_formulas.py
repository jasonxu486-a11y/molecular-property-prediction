import pandas as pd
import pickle

# Load boiling point data
bp_df = pd.read_csv(r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\沸点预测\模型数据\clean_data.csv', encoding='utf-8-sig')
print('沸点数据集:')
print(f'总分子数: {len(bp_df)}')
print(f'唯一化学式数: {bp_df["FORMULA"].nunique()}')

# Group by formula and show examples
print('\n按化学式分组示例:')
for formula in bp_df['FORMULA'].unique()[:10]:
    group = bp_df[bp_df['FORMULA'] == formula]
    print(f'{formula}: {len(group)} 个分子')
    for _, row in group.iterrows():
        print(f'  - {row["NAME"]} (CAS: {row["CAS No"]})')

# Load solubility data
print('\n\n水溶解度数据集:')
sol_df = pd.read_csv(r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\水溶解度预测\模型数据\logs_dataset_v8.csv', encoding='utf-8-sig')
print(f'总分子数: {len(sol_df)}')

# Show some examples with same formula
print('\n化学式重复示例:')
from collections import Counter
formula_counts = Counter(sol_df['SMILES'])
for smiles, count in formula_counts.most_common(10):
    if count > 1:
        print(f'{smiles}: {count} 次')
