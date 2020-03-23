from demo20200322_CollaborativeFiltering.cf.util import read_train_data

# 1. 获取用户和用户之间的相似度
# 1.1 正常逻辑的用户相似度计算

def user_normal_simmilarity(train_data):
        # 相似度字典
        w = dict()
        for u in train_data.keys():
            if w.get(u, -1) == -1:
                w[u] = dict()
            for v in train_data.keys():
                if u == v: continue
                # 相似度计算，通过两个用户共同拥有的物品集合数量
                w[u][v] = len(set(train_data[u]) & set(train_data[v]))  # jaccard distance
                w[u][v] = 2 * w[u][v] / (len(train_data[u]) + len(train_data[v]))
        return w
        # # print(w['196'])
        # # 发现相似度为0的数据  '826': 0.0  O(n^2)
        # print('all user cnt: ', len(w.keys()))
        # print('user_196 sim user cnt: ', len(w['196']))
        # print(sorted(w['196'].items(),key=lambda x:x[1],reverse=False)[:10])


# 1.2 优化计算用户与用户之间的相似度 user->item => item->user

def user_sim(train_data):
    # 建立item->users的倒排表
    item_users = dict()
    for u, items in train_data.items():  # items item,rating
        for i in items.keys():
            if item_users.get(i, -1) == -1:
                item_users[i] = set()
            item_users[i].add(u)

    # 计算共同的items数量
    C = dict()

    for i, users in item_users.items():
        for u in users:
            if C.get(u, -1) == -1:
                C[u] = dict()
            for v in users:
                if u == v:
                    continue
                if C[u].get(v, -1) == -1:
                        C[u][v] = 0
                C[u][v] += 1
    del item_users  # 从内存中删除
    for u,sim_users in C.items():
        for v,cuv in sim_users.items():
            C[u][v] = 2*C[u][v]/(float(len(train_data[u])+len(train_data[v])))
    return C
    # # print(C['196'])
    # # 发现相似度为0的数据  '826': 0.0  O(n^2)
    # print('all user cnt: ', len(C.keys()))
    # print('user_196 sim user cnt: ', len(C['196']))
    # print(sorted(C['196'].items(),key=lambda x:x[1],reverse=False)[:10])


def recommend(user,train_data,C,k=5):
    rank = dict()
    # 用户之前评论过的电影
    watched_items = train_data[user].keys()
    # 取相似的k个用户的items
    for v,cuv in sorted(C[user].items(),key=lambda x:x[1],reverse=True)[:k]:
        # 取相似k个用户的items ②rating
        for i,rating in train_data[v].items():
            # 过滤掉已经评论过的电影（购买过的商品）
            if i in watched_items:
                continue
            elif rank.get(i,-1) == -1:
                rank[i] = 0
            # 物品的打分是①用户相似度[0-1]*②相似用户对电影的打分[0-5]=[0-1]
            # 相似用户评论了同一个物品是累加操作
            rank[i] += cuv * rating
    return rank


if __name__ == '__main__':
    train_data = read_train_data()
    # user_normal_simmilarity(train_data)
    C = user_sim(train_data)
    user_id = '196'
    rank = recommend(user_id,train_data,C)
    print(sorted(rank.items(),key=lambda x:x[1],reverse=True)[:10])
    # [('100', 6.241226030138492), ('204', 4.6451259221991), ('211', 4.143786182755323), ('56', 4.13879680827505), ('603', 4.041840955842541)]