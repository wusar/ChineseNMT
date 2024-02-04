import pandas as pd
from sklearn.model_selection import train_test_split

# 读取TSV文件
file_path = './data/news/news-commentary-v15.en-zh.tsv'
data = pd.read_csv(file_path, sep='\t')

# 首先，划分数据为训练集和剩余部分（测试集+验证集）
train_data, remaining_data = train_test_split(data, test_size=0.1, random_state=42)

# 然后，将剩余部分再划分为测试集和验证集
test_data, valid_data = train_test_split(remaining_data, test_size=0.5, random_state=42)

# 输出划分后数据的大小
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")
print(f"验证集大小: {len(valid_data)}")

# 将划分后的数据集写回TSV文件
train_data.to_csv('./data/splited/train.tsv', sep='\t', index=False)
test_data.to_csv('./data/splited/test.tsv', sep='\t', index=False)
valid_data.to_csv('./data/splited/valid.tsv', sep='\t', index=False)