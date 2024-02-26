'''
    数据处理
    将训练集、验证集、测试集的数据（input, output）处理为（src, score）
    'APE_test_set.csv', 'APE_train_set.csv', 'APE_valid_set.csv'
    {input 0 / output 1}
'''


import pandas as pd

datasets = ['APE_test_set.csv', 'APE_train_set.csv', 'APE_valid_set.csv']
for dataset in datasets:

    df = pd.read_csv(dataset)
    name = dataset.split('_')[1]

    new_df = pd.DataFrame(columns=["src", "score"])

    for index, row in df.iterrows():
        new_df = pd.concat([new_df, pd.DataFrame({"src": [row["input"]], "score": [0]})], ignore_index=True)
        new_df = pd.concat([new_df, pd.DataFrame({"src": [row["output"]], "score": [1]})], ignore_index=True)

    # print(new_df)
    new_df.to_csv("APE_score_{}.csv".format(name), index=False)
    print("{} set ready!".format(name))
