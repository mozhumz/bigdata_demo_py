'''
基于物品的协同过滤算法给用户推荐那些和他们之前喜欢的物品相似的物品。
ItemCF算法并不利用物品的内容属性计算物品之间的相似度，它主要通过分析用户的行为记录计算物品之间的相似度。
该算法认为，物品A和物品B具有很大的相似度是因为喜欢物品A的用户大都也喜欢物品B
N(i)&N(j)/N(i)
分母 N(i) 是喜欢物品i的用户数，而分子 N(i)&N(j) 是同时喜欢物品i和物品j的用户，
上述公式可以理解为喜欢物品i的用户中有多少比例的用户也喜欢物品j，当喜欢物品i和物品j的人群相似度越高，两个物品越相似
'''
from demo20200322_CollaborativeFiltering.hyj.util import get_train_data
import math

# train_data(k=userId v=dict(k=item,v=rating))
train_data = get_train_data()
# k=itemId v=喜欢电影i的人数
N = dict()
# k=itemId1 v=dict(k=itemId2 v=同时喜欢item1 item2的人数)
C = dict()

for u, items in train_data.items():
    for i in items.keys():
        if N.get(i, -1) == -1:
            N[i] = set()
        #     喜欢电影i的人数
        N[i].add(u)
        if C.get(i, -1) == -1:
            C[i] = dict()

        for j in items.keys():
            if i == j: continue
            if C[i].get(j, -1) == -1:
                C[i][j] = 0
                # 同时喜欢item1 item2的人数
            C[i][j] += 1

# 计算电影i和j的相似度
for i, items in C.items():
    for j in items:
        C[i][j] = float(C[i][j]) / len(N[i] | N[j])

# 根据用户喜欢的物品，查找前k个相似物品
k = 50
uid = '385'
map=train_data[uid]
items = map.items()
print(items)
print("--------------------------------------")

# 相似电影集合 k=itemId v=相似度
reco = dict()
for i, rating in items:
    # 获取item的相似物品
    if C.get(i, -1) == -1:
        continue
    for j in C[i].keys():
        if j in map.keys(): continue
        if reco.get(j, -1) == -1:
            reco[j] = 0.0
        reco[j] += (C[i][j] * rating)


print(sorted(reco.items(), key=lambda x: x[1], reverse=True)[:k])

# [('69', 183.3237202232321), ('179', 180.91082330946145), ('196', 179.2279600694895), ('202', 176.44636140456382),
# ('96', 175.9423151702241), ('64', 175.0024976417742),