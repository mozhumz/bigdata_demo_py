import pandas
from demo20200322_CollaborativeFiltering.hyj.util import mid_train_data_path

src_data = '../data/u.data'

df = pandas.read_csv(src_data, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
# print(df.head())
# train_data 二位字典表 第一层key为用户id，value为字典（key=电影id，value=电影打分）
train_data = dict()
for _, row in df.iterrows():
    user_id = str(row['user_id'])
    item_id = str(row['item_id'])
    rating = row['rating']

    if train_data.get(user_id, -1) == -1:
        train_data[user_id] = {item_id: rating}
    else:
        train_data[user_id][item_id] = rating

with open(mid_train_data_path,encoding='utf-8',mode='w') as f:
    f.write(str(train_data))
    print('ok')