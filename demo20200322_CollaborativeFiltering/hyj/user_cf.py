from demo20200322_CollaborativeFiltering.hyj.util import get_train_data

# 1. 获取用户和用户之间的相似度
'''
用户相似度
# same_count=用户间共同看过的电影数量 u_count=用户u看过的电影总数 v_count=用户v看过的电影总数
# sim（用户相似度）=same_count/((u_count+v_count)/2)
电影推荐：
给定用户u，根据用户相似度找到前k个用户
遍历相似用户，获取相似用户v的看过的电影列表m_list，过滤掉用户u看过的电影
电影i打分 +=用户v的相似度*v对电影i的打分
'''
'''
正常逻辑的用户相似度计算
'''
train_data = get_train_data()
# key=用户id value=dict（k=用户id v=用户相似度）
sim_dic = dict()
sim_dic2 = dict()


def normal_user_sim():
    for u in train_data.keys():
        if sim_dic.get(u, -1) == -1:
            sim_dic[u] = dict()
        for v in train_data.keys():
            if u == v: continue
            if sim_dic[u].get(v, -1) == -1:
                sim_dic[u][v] = 0
            sim_dic[u][v] = 2 * len(set(train_data[u]) & set(train_data[v])) / (len(train_data[u]) + len(train_data[v]))


# 1.1 正常逻辑的用户相似度计算
# normal_user_sim()

# 1.2 优化计算用户与用户之间的相似度 user->item => item->user
# key=电影id value=用户id集合
item_users = dict()
for u, items in train_data.items():
    for i in items.keys():
        if item_users.get(i, -1) == -1:
            item_users[i] = set()
        item_users[i].add(u)
# sim_dic2 计算相同的电影数量
for i, users in item_users.items():
    for u in users:
        if sim_dic2.get(u, -1) == -1:
            sim_dic2[u] = dict()
        for v in users:
            if u == v: continue
            if sim_dic2[u].get(v, -1) == -1:
                sim_dic2[u][v] = 0

            sim_dic2[u][v] += 1

# for users in train_data.keys():
#     for u in users:
#         for v in users:
#             if u == v: continue
#             if sim_dic2[u].get(v, -1) == -1: continue
#             sim_dic2[u][v]=sim_dic2[u][v]*2/(len(train_data[u])+len(train_data[v]))
# sim_dic2 根据相同电影数计算相似度
for u, sim_users in sim_dic2.items():
    for v, cnt in sim_users.items():
        sim_dic2[u][v] = sim_dic2[u][v] * 2 / (len(train_data[u]) + len(train_data[v]))

# 给定用户，根据相似用户推荐top-n的电影
uid = '196'
# 获取前k个相似用户
k = 5
# k=用户id v=用户相似度
sim_users = sorted(sim_dic2[uid].items(), key=lambda x: x[1], reverse=True)[:k]

# 用户已经看过的电影
watched = train_data[uid].keys()
rank = dict()
for u, sim in sim_users:
    for i in train_data[u].keys():
        if i in watched: continue
        if rank.get(i, -1) == -1: rank[i]=0
        rank[i] += sim * train_data[u][i]

print(sorted(rank.items(),key=lambda x:x[1],reverse=True)[:k])