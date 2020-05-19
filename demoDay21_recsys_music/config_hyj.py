import pandas as pd
import os
# 存储一切对应代码的配置信息，主要的配置信息在这个文件中进行修改

# 原始数据路径
data_path = 'G:\\idea_workspace\\bigdata\\bigdata_demo_py\\demoDay21_recsys_music\\data\music_data'
# 中间数据路径
music_mid_data_path = "data/music_mid_data_hyj"

# 读取音乐源数据
def music_data(nrows=None):
    music_meta=os.path.join(data_path,'music_meta')
    data=pd.read_csv(music_meta,nrows=nrows,sep='\001',names=['item_id','item_name','desc','total_timelen','location','tags'],
                dtype={'item_id':str,'item_name':str,'desc':str,'total_timelen':str,'tags':str})

    data.fillna('-')
    del data['desc']
    return data

def user_profile(nrows=None):
    user_meta=os.path.join(data_path,'user_profile.data')
    return pd.read_csv(user_meta,nrows=nrows,sep=',',names=['user_id','gender','age','salary','province'])

# print(user_profile(10))

def user_watch(nrows=None):
    data=os.path.join(data_path,'user_watch_pref.sml')
    pd.read_csv(data,nrows=nrows,names=[],sep='')
