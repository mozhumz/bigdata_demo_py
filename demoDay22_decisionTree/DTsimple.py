import math


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    cols = ['no surfacing', 'flippers']

    return data_set, cols

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

#  计算熵
def calc_ent(data_set):
    n = len(data_set)
    label_cnt = {}
    for featVec in data_set:
        cur_label = featVec[-1]
        if cur_label not in label_cnt.keys():
            label_cnt[cur_label] = 0
        label_cnt[cur_label] += 1

    E = 0.0
    for key in label_cnt:
        prob = float(label_cnt[key]) / n
        # 以e为底
        E -= prob * math.log(prob)
    return E


# 划分数据集：根据特征每个值划分数据集  i:age,value:青年/中年/老年
def split_data_set(data_set, i, value):
    ret_data_set = []
    for featVec in data_set:
        if featVec[i] == value:
            # 将i之前和之后的特征重新放入样本中，把i特征去掉[1,1,yes]=>[[1][1,yes]]
            # 1[][]
            reduced_feat_vec = featVec[:i]  # 切片时不包含i的数据，i之前的数据
            reduced_feat_vec.extend(featVec[i + 1:]) # i之后的数据
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


# 选择最好的特征进行分裂
def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_ent(data_set)  # 第一层的熵
    best_info_gain = 0.0
    best_feature = -1
    # i相当于年龄
    for i in range(num_features):
        # 获取当前年龄这一列的所有值，为了下一步获取青年，中年，老年三个唯一值
        feat_list = [example[i] for example in data_set]
        # 获取青年，中年，老年。三个不同值的唯一值
        unique_vals = set(feat_list)
        new_entropy = 0.0
        # value相当于青年，中年，老年
        for value in unique_vals:
            # 比如属于青年的数据集合
            sub_data_set = split_data_set(data_set, i, value)
            # 青年中年老年在总（上一次划分数据集）样本中的占比
            prob = len(sub_data_set) / float(len(data_set))
            # 划分之后的数据集的信息entropy
            new_entropy += prob * calc_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 叶子结点返回的规则：样本中类别数量最多的一个类别
def majority_cnt(class_list):
    d = dict()
    for c in class_list:
        if d.get(c, -1) == -1:
            d[c] = 0
        d[c] += 1
    return max(d.items(), key=lambda x: x[1])[0]


# 生成决策树，递归
def create_tree(data_set, cols):
    class_list = [example[-1] for example in data_set]
    # 判断类别列表中是不是只有一个值，如果是，表示已经是“纯”了  [1,1,1,1,1] 1:5
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果数据维度为1，要基于最后一个特征划分，相当于接下来没有特征了，特征列表为空
    if len(data_set[0]) == 1:
        # 返回样本中类别数量最多的一个类别
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = cols[best_feat]

    # 初始化树，用字典作为树
    my_tree = {best_feat_label: {}}
    del (cols[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    # 用特征值划分数据集
    for value in unique_vals:
        subcols = cols[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), subcols)
    return my_tree


if __name__ == '__main__':
    data_set, cols = create_data_set2()
    treeModel = create_tree(data_set, cols)
    print(treeModel)
#     {'age': {'老年': {'salary': {'高': 'yes', '低': 'no', '中': 'no'}}, '中年': {'salary': {'高': 'yes', '低': 'no', '中': 'no'}}, '青年': 'yes'}}
