import os
import pandas as pd

data_path = 'D:\data\data'
data_file = os.path.join(data_path,'orders.csv')
cols = ["order_id","user_id","eval_set","order_number","order_dow","hour","day"]

df = pd.read_csv(data_file)
df = df.fillna(0)
# print(df.head())
# print(df.columns)

for _,row in df.iterrows():
    print(row)
    print(row['order_id'])
    # d = {}
    # for col in df.columns:
    #     d[col] = row[col]
    # print(d)
    break




