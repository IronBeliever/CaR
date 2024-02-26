'''
    将语料分为 训练集、验证集、测试集
    csv 格式 
    固定各个数据集，对比看方法效果
'''

import re
import Levenshtein
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def replace_unicode_escape(sequence):
    # 将字符串中所有的"\uxxxx"替换成“\uxxxx”，其中x代表数字或字母
    pattern = r'\\u([0-9a-fA-F]{4})'
    replacement = lambda match:chr(int(match.group(1), 16))
    return re.sub(pattern, replacement, sequence)



with open("聊天数据集_APE_1234-4985.json", "r") as f:
    data = json.load(f)

data_pair = []
for item in data:
    data_pair.append({"input":item["input"], "output":item["output"]})

distances = []
for dic in data_pair:
    distances.append(Levenshtein.distance(dic['input'], dic['output']))

combined = zip(distances, [dic['input'] for dic in data_pair], [dic['output'] for dic in data_pair])
sorted_combined = sorted(combined, key=lambda x:x[0], reverse=False)
sorted_distances, sorted_raw, sorted_improved = zip(*sorted_combined)
# cnt, dist,_ = plt.hist([ele for ele in sorted_distances], bins=300)
# data_2301 = [{"input":tup[1], "output":tup[2]} for tup in sorted_combined[1211:]]
# print(len(data_2301))

_, input, output = zip(*sorted_combined[1211:])

data = pd.DataFrame({"input": input, "output": output})
data.to_csv("APE_dataset.csv", index=False, header=True)

train, valid_test = train_test_split(data, test_size=0.2, random_state=42)
valid, test = train_test_split(valid_test, test_size=0.5, random_state=42)

train.to_csv('APE_train_set.csv', columns=['input', 'output'], index=False, header=True)
valid.to_csv('APE_valid_set.csv', columns=['input', 'output'], index=False, header=True)
test.to_csv('APE_test_set.csv', columns=['input', 'output'], index=False, header=True)
