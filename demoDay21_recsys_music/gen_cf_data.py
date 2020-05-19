import demoDay21_recsys_music.config as conf

train_file = conf.train_file


def user_item_score(action_num=100, tag='sum'):
    '''
    将原始数据处理成cf的输入数据，类似udata中的数据 user_id,item_id,rating
    :return: data(DataFrame)[user_id,item_id,score]
    '''
    # 获取用户行为数据
    df_user_watch = conf.user_watch(action_num)

    # 思考怎么获取到score：stay_seconds/音乐的total_timelen,听这个音乐的占比
    # stay_seconds在user_watch，total_timelen在music_meta
    df_music_meta = conf.music_data()
    # pandas里的merge相当于hive里面join
    data = df_user_watch.merge(df_music_meta, how='inner', on='item_id')
    # 从内存中删除基础表数据
    del df_user_watch
    del df_music_meta
    # pandas apply 相当于spark rdd map操作 score = 20s / 300s pandas官网简单使用说明，spark官网
    data['score'] = data.apply(lambda x: float(x['stay_seconds']) / float(x['total_timelen']), axis=1)
    # 选要使用的列，和udata数据类型一致
    data = data[['user_id', 'item_id', 'score']]
    # 由于（user_id，item_id）在不同时间段会有重复，所以需要将相同的合并，[sum]，avg
    if tag == 'sum':
        data = data.groupby(['user_id', 'item_id'])['score'].sum().reset_index()
    elif tag == 'avg'or'mean':
        data = data.groupby(['user_id', 'item_id']).score.mean().reset_index()
    return data


def train_from_df(df, col_name=['user_id', 'item_id', 'score']):
    '''
    将DataFrame数据处理成cf输入的数据形式（dict）
    :param df: DataFrame数据
    :param col_name: 对应所需要取到的列名数组
    :return: 最终dict数据
    '''
    d = dict()
    # 按行遍历 _表示索引index
    for _, row in df.iterrows():
        user_id = str(row[col_name[0]])
        item_id = str(row[col_name[1]])
        rating = row[col_name[2]]
        if d.get(user_id, -1) == -1:
            d[user_id] = {item_id: rating}
        else:
            d[user_id][item_id] = rating
    return d


if __name__ == '__main__':
    # 控制用户数据量，在计算用户与用户相似度
    data = user_item_score(50000)
    train = train_from_df(data)
    del data
    # 将训练数据存储起来，这样下次使用直接读取不需要再处理
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(str(train))
