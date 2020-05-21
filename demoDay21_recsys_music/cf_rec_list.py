import demoDay21_recsys_music.cf.user_cf as uc
import demoDay21_recsys_music.cf.item_cf as ic
import demoDay21_recsys_music.config as conf

UCF_PREFIX = conf.UCF_PREFIX
ICF_PREFIX = conf.ICF_PREFIX

# 推荐结果输出路径：user_base+item_base中间结果
cf_rec_lst_outfile = conf.cf_rec_lst_outfile

# 读取上一步gen_cf_data的train data
with open(conf.train_file,'r',encoding='utf-8') as f:
    train = eval(f.read())
print('CF train  data have loaded! Start compute user similarity ...')
# print(train)

reclst = dict()

'''
user base
'''
# 计算用户与用户的相似度矩阵并存储
user_user_sim = uc.user_sim(train)
print('Compute done! Saving user-user similarity matrix ...')
with open(conf.user_user_sim_file, 'w') as fw:
    fw.write(str(user_user_sim))

# 对每个用户计算推荐物品集合 recall物品2
for user_id in train.keys():
    rec_item_list = uc.recommend(user_id,train,user_user_sim,10)
    # 标识从哪个离线召回策略出来，主要原因是不同的策略会有相同的user_id
    user_id = UCF_PREFIX + user_id
    reclst[user_id] = sorted(rec_item_list.items(),
                             key=lambda x: x[1],
                             reverse=True)[0:20]
print('User base done! Item base starting ...')

del user_user_sim

'''
item base
'''
# 计算歌曲与歌曲的相似度矩阵并存储
item_item_sim = ic.item_sim(train)
with open(conf.item_item_sim_file, 'w') as ifw:
    ifw.write(str(item_item_sim))

# 对每个用户计算推荐物品集合 recall物品1
for user_id in train.keys():
    rec_item = ic.recommendation(train,user_id,C=item_item_sim,k=10)
    user_id = ICF_PREFIX + user_id
    reclst[user_id] = sorted(rec_item.items(),
                             key=lambda x: x[1],
                             reverse=True)[0:]
del item_item_sim

# 将user_cf和item_cf的推荐列表存储起来
with open(cf_rec_lst_outfile,'w',encoding='utf-8') as wcf:
    wcf.write(str(reclst))