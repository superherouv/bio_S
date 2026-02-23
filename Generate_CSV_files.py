import os
import random
import pandas as pd

random.seed(53)  # 保证结果可重复

# 文件夹路径
base_dir = './dataset/ETH3x100/'
categories = ['apple', 'car', 'cup']  # 类别名称请按实际修改

# 存储所有记录
train_records = []
val_records = []
test_records = []

for label, category in enumerate(categories):
    category_dir = os.path.join(base_dir, category)
    all_files = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.png'))]
    all_paths = [os.path.join(category_dir, f) for f in all_files]

    random.shuffle(all_paths)
    train = all_paths[:70]
    val = all_paths[70:80]
    test = all_paths[80:]

    train_records.extend([(p, label) for p in train])
    val_records.extend([(p, label) for p in val])
    test_records.extend([(p, label) for p in test])

# 转成 DataFrame 并保存 CSV
for records, name in zip([train_records, val_records, test_records],
                         ['Species_train_annotation.csv', 'Species_val_annotation.csv', 'Species_test_annotation.csv']):
    df = pd.DataFrame(records, columns=['path', 'species'])
    df.to_csv(name)
