import math


def create_data_set2():
    data_set = [['青年', '高', 'yes'],
                ['青年', '高', 'yes'],
                ['青年', '中', 'yes'],
                ['青年', '中', 'yes'],
                ['青年', '中', 'yes'],
                ['青年', '中', 'yes'],
                ['青年', '中', 'yes'],
                ['青年', '低', 'yes'],
                ['青年', '低', 'yes'],
                ['中年', '高', 'yes'],
                ['中年', '高', 'yes'],
                ['中年', '高', 'no'],
                ['中年', '中', 'no'],
                ['中年', '中', 'no'],
                ['中年', '中', 'no'],
                ['中年', '低', 'no'],
                ['老年', '高', 'yes'],
                ['老年', '中', 'no'],
                ['老年', '低', 'no'],
                ['老年', '低', 'no'],
                ]
    cols = ['age', 'salary']

    return data_set, cols


'''
根据样本计算信息熵
'''


def get_entropy_by_data(data_list, feat_idx):
    # 计算分类的信息熵 p=Dk/D
    D = len(data_list)
    # y_dict = dict()
    # v=[0.0,{y:count}]
    feat_dict = dict()
    for row in data_list:
        row_feat_val = row[feat_idx]
        # y_key = '-1_' + row[-1]
        feat_key = str(feat_idx) + '_' + str(row[feat_idx])
        # set_dict_count(y_dict, y_key)
        set_dict_count(feat_dict, feat_key, row[-1])
    # 样本分类信息熵
    # h_d = get_entropy_by_dict(D, y_dict.items())
    # 特征分类信息熵
    res = 0.0
    for key, val in feat_dict.items():
        Di = val[0]
        res += Di / D * get_entropy_by_list(Di, val[1].values())

    return res


def set_dict_count(y_dict, y_key, y):
    # 统计Di
    if y_dict.get(y_key, -1) == -1:
        y_dict[y_key] = [0.0, {}]
    y_dict[y_key][0] += 1.0
    # 统计Dik
    if y_dict[y_key][1].get(y, -1) == -1:
        y_dict[y_key][1][y] = 0.0
    y_dict[y_key][1][y] += 1.0


'''
获取分类y的信息熵
'''


def get_y_entropy(data_list):
    res = 0.0
    y_dict = dict()
    for row in data_list:
        if y_dict.get(row[-1], -1) == -1:
            y_dict[row[-1]] = 0
        y_dict[row[-1]] += 1.0
    D = len(data_list)
    return get_entropy_by_list(D, y_dict.values())


'''
根据字典分类获取信息熵
'''


def get_entropy_by_list(D, Dk_list):
    h_d = 0.0
    for Dk in Dk_list:
        p = Dk / D
        h_d -= p * math.log2(p)
    return h_d


'''
根据样本和特征获取信息增益最大的特征索引
'''


def get_best_feat_idx(data_list):
    row0 = data_list[0]
    y_idx = len(row0) - 1
    res_idx = -1
    res_gain = 0.0
    y_entropy = get_y_entropy(data_list)
    for idx in range(len(row0)):
        if y_idx == idx:
            continue
        feat_entropy = get_entropy_by_data(data_list, idx)
        feat_gain = y_entropy - feat_entropy
        if feat_gain > res_gain:
            res_gain = feat_gain
            res_idx = idx
    return res_idx


'''
根据样本和特征取值，筛选指定特征取值的数据 如年龄中的青年数据
'''


def get_feat_data(data_list, idx, val):
    res = []
    for row in data_list:
        if row[idx] == val:
            res_row = row[:]
            # 去掉特征列
            del (res_row[idx])
            res.append(res_row)
    return res


'''
获取list中的众数
'''
def get_most_element(data_list):
    data_dict = dict()
    for i in data_list:
        if data_dict.get(i, -1) == -1:
            data_dict[i] = 0
        data_dict[i] += 1
    # max_count=0
    # res_key=''
    # for key,count in data_dict.items():
    #     if count>max_count:
    #         max_count=count
    #         res_key=key
    sort_list=sorted(data_dict.items(),key=lambda item:item[1],reverse=True)
    # 返回第一个item中的key
    return sort_list[0][0]


'''
创建决策树
'''


def create_tree(data_list, cols):
    # 判断数据是否纯净
    class_list = [row[-1] for row in data_list]
    if (len(set(class_list)) == 1):
        return class_list[0]
    # 特征如果只剩一维，则无特征可划分，选取样本中y的众数作为分类结果 [1,1,1,2,2]=>1
    if len(data_list[0])==1:
        return get_most_element(class_list)
    # 获取信息增益最大的特征
    best_feat_idx=get_best_feat_idx(data_list)
    best_feat_name=cols[best_feat_idx]
    # 从特征列表删除该特征
    del(cols[best_feat_idx])
    # 初始化树
    tree = {best_feat_name: {}}
    # 获取该特征的所有取值列表
    feat_val_list=[row[best_feat_idx] for row in data_list]
    unique_val_list=set(feat_val_list)

    # 遍历该特征的取值 获取属于该特征的样本 递归建树
    for idx,val in enumerate(unique_val_list):
        # 复制剩余的特征列表
        subcols=cols[:]
        tree[best_feat_name][val]=create_tree(get_feat_data(data_list,best_feat_idx,val),subcols)
    return tree


if __name__ == '__main__':
    data_list, cols = create_data_set2()
    # res=get_entropy_by_data(data_list,0)
    # res1=get_entropy_by_data(data_list,1)
    # print(res)
    # print(res1)
    # get_best_feat_idx(data_list)
    # print(get_feat_data(data_list, 0, '青年'))
    # print(get_most_element([1,1,1,2,2,2,2,3,3,3,3,3]))
    # list1=[1,1,1,2,2,2,2,3,3,3,3,3]
    # for idx,val in enumerate(list1):
    #     print(idx)
    tree=create_tree(data_list,cols)
    print(tree)
    # {'age': {'中年': {'salary': {'低': 'no', '高': 'yes', '中': 'no'}}, '老年': {'salary': {'低': 'no', '高': 'yes', '中': 'no'}}, '青年': 'yes'}}