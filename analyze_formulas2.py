import pandas as pd
from collections import Counter

# Load boiling point data
bp_df = pd.read_csv(r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\沸点预测\模型数据\clean_data.csv', encoding='utf-8-sig')

# Count formulas
formula_counts = Counter(bp_df['FORMULA'])

# Show formulas with multiple molecules
print('化学式重复示例（沸点数据集）:')
for formula, count in formula_counts.most_common(20):
    if count > 1:
        print(f'{formula}: {count} 个分子')
        group = bp_df[bp_df['FORMULA'] == formula]
        for _, row in group.iterrows():
            print(f'  - {row["NAME"]} (CAS: {row["CAS No"]})')
