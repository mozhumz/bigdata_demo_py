import demoDay21_recsys_music.hyj.config_hyj as conf
import pandas as pd
import  math
import common.common_util as util

a = 0.6
user_id = '010af058c9e6aa1109de610cae30fdf8'
# 获取该用户听过的歌曲 item_name
music_df=conf.music_data()
user_watch_df=conf.user_watch()
user_merge_df=user_watch_df.merge(music_df,how='inner',on='item_id')
user_df=user_merge_df.loc[user_merge_df['user_id']==user_id,['item_name']]
pd.unique(user_df['item_name'])

# 读取用户离散特征
user_feat_dict=dict()
with open(conf.user_feat_map_file,mode='r',encoding='utf-8') as f:
    user_feat_dict=eval(f.read())

# 读取用户连续特征
user_cross_feat_dict=dict()
with open(conf.cross_file,mode='r',encoding='utf-8') as f:
    user_cross_feat_dict=eval(f.read())

# 读取训练好的模型参数
model_dict=dict()
with open(conf.model_file,mode='r',encoding='utf-8') as f:
    model_dict=eval(f.read())

W=model_dict['W']
b=model_dict['b']

# 读取离线状态-该用户的推荐物品 ucf icf
rec_item_all=dict()

with open(conf.cf_rec_lst_outfile,mode='r',encoding='utf-8') as f:
    rec_lst=eval(f.read())

'''ucf的推荐物品'''
ucf_key=conf.UCF_PREFIX+user_id
ucf_lst=rec_lst[ucf_key]

for item_id,score in ucf_lst:
    rec_item_all[item_id]=[float(score)*a,'user_base']

'''icf的推荐物品'''
icf_key=conf.ICF_PREFIX+user_id
icf_lst=rec_lst[icf_key]

for item_id,score in icf_lst:
    if rec_item_all.get(item_id,-1)==-1:
        rec_item_all[item_id]=[float(score)*(1-a),'item_base']
    else:
        rec_item_all[item_id][0]+=float(score)*(1-a)
        rec_item_all[item_id][1]='user+item'

# 获取用户特征的index
user_info_df=conf.user_profile()
age,gender,salary,province='','','',''
for _,row in user_info_df.loc[user_info_df['user_id']==user_id].iterrows():
    age,gender,salary,province=row['age'],row['gender'],row['salary'],row['province']
    (age_idx,gender_idx,salary_idx,province_idx)=(user_feat_dict['age_'+age],user_feat_dict['gender_'+gender],
                                                  user_feat_dict['salary_'+salary],user_feat_dict['province_'+province])
    print(age,gender,salary,province)



res_list=[]
# 整合icf和ucf推荐的物品，利用LR model打分排序
for item_id in rec_item_all.keys():
    location,item_name='',''
    for _,row in music_df.loc[music_df['item_id']==item_id,:].iterrows():
        location=row['location']
        if(type(location)==float):
            location=None
        item_name=row['item_name']
    print(location,item_name)

    location_idx=None

    if location :
        location_idx=user_feat_dict['location_'+location]


    # 召回的物品打分
    score=rec_item_all[item_id][0]

    # LR-离散特征-打分
    w_score=W[age_idx]+W[gender_idx]+W[salary_idx]+W[province_idx]
    if location_idx:
        w_score+=W[location_idx]
    # LR-连续特征-打分
    uk=user_id+'_'+item_id
    cross_score=float(user_cross_feat_dict.get(uk,0))
    w_score+=W[-1]*cross_score
    w_score+=float(b)

    # sigmoid转换
    sig_score=1/(1+math.exp(-w_score))
    final_score=score*sig_score

    res_list.append((item_id,item_name,final_score,rec_item_all[item_id][1]))

res_file=conf.res_file
util.mkdirs(res_file)
with open(res_file,mode='w',encoding='utf-8') as f:
    f.write(str(res_list))
# 排序
res_sort_list=sorted(res_list,key=lambda x:x[2],reverse=True)
print(res_sort_list)

with open(res_file,mode='r',encoding='utf-8') as f:
    res_sort_list=eval(f.read(res_sort_list))

# topN
filter_lst=res_sort_list[:5]


res=['=>'.join([item_id,item_name,str(final_score),explain]) for item_id,item_name,final_score,explain in filter_lst]
print(res)