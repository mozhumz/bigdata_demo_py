import pandas as pd
from cf.util import mid_train_data_path
'''
user_id item_id rating timestamp
196	242	3	881250949
处理训练数据-> dict{user_id:{item_id:rating},user_id1:{item_id2:rating}}
'''
df =pd.read_csv('../data/u.data',
                sep='\t',
                # nrows=10,
                names=['user_id','item_id','rating','timestamp'])
# print(df.head())
train_data = dict()
for _,row in df.iterrows():
    # print(row)
    user_id = str(row['user_id'])
    item_id = str(row['item_id'])
    rating = row['rating']

    if train_data.get(user_id, -1) == -1:
        train_data[user_id] = {item_id: rating}
    else:
        train_data[user_id][item_id] = rating

# print(train_data)
with open(mid_train_data_path,'w',encoding='utf-8') as f:
    f.write(str(train_data))