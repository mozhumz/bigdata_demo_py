import demoDay21_recsys_music.config as conf
import pandas as pd
import math

a = 0.6
user_id = '010af058c9e6aa1109de610cae30fdf8'
# 以前听什么歌：
user_watch = conf.user_watch()
music_df = conf.music_data()
df = user_watch.merge(music_df, how='inner', on='item_id')

del user_watch
del music_df

# 当前用户听过的音乐，做一个比较：他历史听的音乐和我们推荐的音乐
# df.loc[index, columns_name]
df = df.loc[df['user_id'] == user_id, ['item_name']]
print(pd.unique(df['item_name']))

# step1: 载入特征处理
# load user and item category feature 离散特征map
with open(conf.user_feat_map_file, 'r', encoding='utf-8') as f:
    category_feat_dict = eval(f.read())

# load cross feature 交叉特征
with open(conf.cross_file, 'r', encoding='utf-8') as f:
    cross_feat_dict = eval(f.read())

# step 2:载入模型
with open(conf.model_file, 'r', encoding='utf-8') as f:
    model_dict = eval(f.read())
W = model_dict['W']
b = model_dict['b']
del model_dict

# step3: match/recall(协同过滤，召回候选集)
rec_item_all = dict()
# 3.1 CF
# 3.1.1 user base
with open(conf.cf_rec_lst_outfile, 'r', encoding='utf-8') as f:
    cf_rec_lst = eval(f.read())
# user base的组合user_id
key = conf.UCF_PREFIX + user_id

# 用户协同召回集合2
ucf_rec_lst = cf_rec_lst[key]  # user_base的推荐候选集
for item,score in ucf_rec_lst:
    # a表示user_base的召回候选集的特征权重0.6，item-base 1-0.6=0.4
    # 后面这个user base主要表示推荐的这个item是通过哪个召回策略出来的，通过这个可以增加推荐解释，
    rec_item_all[item] = [float(score) * a, 'user base']

# for item,score in ucf_rec_lst

# 3.1.2 item base
key = conf.ICF_PREFIX + user_id
icf_rec_lst = cf_rec_lst[key]  # item_base的推荐候选集
for item,score in icf_rec_lst:
    if rec_item_all.get(item,-1) == -1:
        rec_item_all[item] = [float(score)*(1-a),'item_base']
    else:
        # 当两种推荐中物品相同时，求和
        rec_item_all[item][0] += float(score)*(1-a)
        rec_item_all[item][1] = 'user+item'   # 同时两个召回策略中都包含对应的item

# step4: 调用用户和物品的服务（从业务数据库中取属性数据）
# 线上要拼特征的时候需要获取对应特征的值
# 4.1 用户属性
user_df = conf.user_profile()
age, gender, salary, province = '', '', '', ''
for _, row in user_df.loc[user_df['user_id'] == user_id, :].iterrows():
    age, gender, salary, province = row['age'], row['gender'], row['salary'], row['province']
    # 查找对应特征的值所对应的index，因为是one-hot编码，下面四个idx上的值都为1
    (age_idx, gender_idx, salary_idx, province_idx) = (category_feat_dict['age_'+age],
                                                       category_feat_dict['gender_'+gender],
                                                       category_feat_dict['salary_' + salary],
                                                       category_feat_dict['province_' + province])
    # 顺便打印用户的属性信息，方便对推荐结果做比较
    print('age: '+age,'gender: '+gender,'salary: '+salary,'province: ' + province)
del user_df

# 获取item的属性信息，查表找idx
# item_name可以通过结巴分词进行切词做特征，one-hot，在训练集上增加one-hot特征，可以放到lr中训练，线上也需要有切词服务，进行切词找idx
rec_lst = []
item_df = conf.music_data()
for item_id in rec_item_all.keys():
    location, item_name = '',''
    for _,row in item_df.loc[item_df['item_id'] == int(item_id),:].iterrows():
        location, item_name = row['location'],row['item_name']
    location_idx = category_feat_dict['location_'+location]

    # 增加交叉特征
    ui_key = user_id+'_'+item_id
    cross_value = float(cross_feat_dict.get(ui_key,0))

    # 排序预测，应用lr进行模型打分 predict
    # y = sigmoid(-wx-b)

    # wx+b
    wx_score = float(b)
    # wx 离散特征部分（one-hot）x [0,1,0,0,0,1]  w [0.1,0.2,0.3,0.4,0.5,0.6]  0.2+0.6
    wx_score += W[age_idx]+W[gender_idx]+W[salary_idx]+W[province_idx]+W[location_idx]
    # 连续特征wx
    wx_score += W[-1]*cross_value

    # lr score
    final_rec_score = 1/(1+math.exp(-(wx_score)))  # sigmoid套公式
    # 召回的组合score=0.6ucf+0.4ucf
    score = rec_item_all[item_id][0]
    # lr score*0.3 协同cf score*0.7，将lr分值和召回策略的分值做个组合
    # final_rec_score = 0.3*final_rec_score+0.7*score
    final_rec_score =final_rec_score * score
    # 对item做一个组装，方便打印信息
    rec_lst.append((item_id, item_name, final_rec_score, rec_item_all[item_id][1]))

# step 5: 排序
rec_sort_list = sorted(rec_lst, key=lambda x: x[2], reverse=True)

# step 6: top N(取5个)
rec_filter_lst = rec_sort_list[:5]

# step 7:返回+包装（return）
ret_list = ['   =>  '.join([i_id,name,str(score),explain]) for i_id,name,score,explain in rec_filter_lst]
print('\n'.join(ret_list))


