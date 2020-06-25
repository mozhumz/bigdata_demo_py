import demoDay21_recsys_music.hyj.config_hyj as conf
import os
'''生成训练数据 k=user_id v={item_id:score}'''
train_file = conf.train_file

# 获取用户对物品的打分 df[user_id item_id score]
def user_item_socre(nrows=100,tag='sum'):
    # 读取音乐元数据
    music_data=conf.music_data()
    # 读取用户听过的音乐数据
    user_watch=conf.user_watch(nrows)
    #数据关联 获取打分score=(用户听过的时长)stay_seconds/total_timelen(音乐总时长)
    data=user_watch.merge(music_data,how='inner',on='item_id')
    # 从内存删除数据
    del music_data
    del user_watch
    data['score']=data.apply(lambda x:float(x['stay_seconds']/float(x['total_timelen'])),axis=1)
    # 返回的是各列的平均值
    # data.mean(axis=0)
    # 只获取目标字段user_id item_id score
    data=data[['user_id','item_id','score']]
    # 相同用户不同时间可能听过相同的音乐 即user_id item_id有重复，score相加 再平均或求和
    if tag=='sum':
        data=data.groupby(['user_id','item_id'])['score'].sum().reset_index()
    elif tag=='avg' or 'mean':
        data=data.groupby(['user_id','item_id']).score.mean().reset_index()
    return data

#将df数据转为字典dict key=user_id value={key:item_id,value:score}
def train_from_df(df,col=['user_id','item_id','score']):
    data=dict()
    for _,row in df.iterrows():
        user_id=row[col[0]]
        item_id=row[col[1]]
        score=row[col[2]]
        if data.get(user_id,-1)==-1:
            data[user_id]={item_id:score}
        else:
            data[user_id][item_id]=score

    return data

if __name__ == '__main__':
    df_train=user_item_socre(50000)
    train=train_from_df(df_train)
    #将训练数据存储到磁盘

    # 文件上一级目录
    # dir=os.path.abspath(os.path.join(os.path.dirname(train_file),os.path.pardir))
    # print(dir)
    # 文件当前目录
    dir2=os.path.dirname(os.path.abspath(train_file))
    print(dir2)
    if not os.path.exists(dir2):
        os.makedirs(dir2)
    with open(train_file,mode='w',encoding='utf-8') as f:
        f.write(str(train))

    print("ok")


