import demoDay21_recsys_music.cf.user_cf as uc
import demoDay21_recsys_music.cf.item_cf as ic
import demoDay21_recsys_music.hyj.config_hyj as conf
import common.common_util as util

UCF_PREFIX=conf.UCF_PREFIX
ICF_PREFIX=conf.ICF_PREFIX
'''存储推荐物品集合 k=user_id+pre（根据相似用户推荐或相似物品推荐） v={item_id:score}'''
rec_list=dict()

# 读取训练数据 k=user_id v={item_id:score}
with open(conf.train_file,mode='r',encoding='utf-8') as f:
    train=eval(f.read())

print(len(train))

'''user base 根据相似用户获取推荐物品'''
# 计算用户相似度并存储 k=user_id value={v_id:score}
user_user_sim=uc.user_sim(train)
user_user_sim_file=conf.user_user_sim_file
util.mkdirs(user_user_sim_file)
with open(user_user_sim_file,mode='w',encoding='utf-8') as f :
    f.write(str(user_user_sim))
print(len(user_user_sim))

# 根据用户相似度获取推荐物品
for user_id in train.keys():
    rank=uc.recommend(user_id,train,user_user_sim,10)
    user_id=UCF_PREFIX+user_id
    # 对获取的推荐物品排序 取前20
    rec_list[user_id]=sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:20]

del user_user_sim

'''item base 根据相似物品获取推荐物品'''

#获取相似物品数据并存储 k=item_id v={item_id1:score1}
item_item_sim=ic.item_sim(train)
item_item_sim_file=conf.item_item_sim_file
util.mkdirs(item_item_sim_file)
with open(item_item_sim_file,mode='w',encoding='utf-8') as f:
    f.write(str(item_item_sim))

# 根据物品相似度获取推荐物品
for user_id in train.keys():
    rank=ic.recommendation(train,user_id,item_item_sim,10)
    user_id=ICF_PREFIX+user_id
    rec_list[user_id]=sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:20]

del item_item_sim

# 存储物品推荐列表
cf_rec_lst_outfile=conf.cf_rec_lst_outfile
util.mkdirs(cf_rec_lst_outfile)
with open(cf_rec_lst_outfile,mode='w',encoding='utf-8') as f:
    f.write(str(rec_list))




