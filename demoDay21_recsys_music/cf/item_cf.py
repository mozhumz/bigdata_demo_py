# from cf.util import read_train_data

# {user_id:{item_id:rating}}
# train_data = read_train_data()


def item_sim(train_data):
    # 相似物品矩阵
    C = dict()  # {item:{item1:sim_score,item2:sim_score}}
    N = dict()  # {item:用户数}
    # 扫一遍数据集，统计相似物品矩阵的分子部分C，和统计item拥有的user dict N
    for u, items in train_data.items():
        for i in items:
            # item拥有多个用户
            if N.get(i, -1) == -1:
                # N[i] = 0
                N[i] = set()
            # N[i] += 1
            N[i].add(u)
            if C.get(i, -1) == -1:
                C[i] = dict()
            for j in items:
                if i == j:
                    continue
                elif C[i].get(j, -1) == -1:
                    C[i][j] = 0
                C[i][j] += 1

    # 扫一遍相似物品矩阵C，将分母除进去
    for i, sim_items in C.items():
        for j, cij in sim_items.items():
            # C[i][j]= 2*cij/(N[i]+N[j]*1.0)
            C[i][j] = cij / len(N[i] | N[j])
    return C


def recommendation(train_data, user_id, C, k):
    # 存放最终结果
    rank = dict()
    sum_ = dict()
    Ru = train_data[user_id]
    for i, rating in Ru.items():
        # rating = rating_old*func(time)
        for j, sim_score in sorted(C[i].items(), key=lambda x: x[1], reverse=True)[:k]:
            # 过滤这个user已经打过分的item
            if j in Ru:
                continue
            elif rank.get(j, -1) == -1:
                rank[j] = 0

            # Ru中item1相似的集合和item2相似的集合中有相同的item，是分值相加形式
            rank[j] += sim_score * rating
            # sum|cij| 对应ppt sum(|wij|),只有在相同的itemj，不同的itemi进行ppt中多值相加
            if sum_.get(j,-1)==-1:
                sum_[j] = 0
            sum_[j] += sim_score
    for j in rank.keys():
        rank[j] /= sum_[j]
    return rank


# if __name__ == '__main__':
#     C = item_sim(train_data)
#     # print(sorted(C['196'].items(), key=lambda x: x[1], reverse=True)[:10])
#     rank = recommendation(train_data, user_id='196', C=C, k=10)
#     print(sorted(rank.items(), key=lambda x: x[1], reverse=True))
