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

# 读取用户信息
def user_profile(nrows=None):
    user_meta=os.path.join(data_path,'user_profile.data')
    return pd.read_csv(user_meta,nrows=nrows,sep=',',names=['user_id','gender','age','salary','province'],
                       dtype={'user_id':str,'gender':str,'age':str,'salary':str,'province':str})

# print(user_profile(10))

# 读取用户听过的音乐
def user_watch(nrows=None):
    data=os.path.join(data_path,'user_watch_pref.sml')
    return pd.read_csv(data,nrows=nrows,names=['user_id','item_id','stay_seconds','hour'],sep='\001',
                       dtype={'user_id':str,'item_id':str,'stay_seconds':int,'hour':int})

# print(user_watch(10))

# #########路径配置#################
train_file = '%s/train_dict.txt' % music_mid_data_path

# 相似度矩阵存储路径
user_user_sim_file = '%s/sim_data/uu.sim' % music_mid_data_path
item_item_sim_file = '%s/sim_data/ii.sim' % music_mid_data_path

# 最终候选集存储
cf_rec_lst_outfile = '%s/reclst.dict' % music_mid_data_path

# 交叉特征存储
cross_file = '%s/cross_feat.dict' % music_mid_data_path

# one-hot编码映射表
user_feat_map_file = '%s/feat/one_hot.dict' % music_mid_data_path
# model储存
model_file = '%s/models/lr.model' % music_mid_data_path

# 不同召回策略标识
UCF_PREFIX = 'UCF_'
ICF_PREFIX = 'ICF_'