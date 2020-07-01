import pandas as pd
import math
from decimal import  *
'''
计算信息熵 H(D)=-sum(p_k*log(p_k))
p_k=Dk/D Dk:分类为k的样本数
'''
def get_entropy(p_list):
    # h=Decimal('0.0')
    h=0.0
    for p in p_list:
        if p<=0:
            continue
        # str_p=str(p)

        # h+=Decimal(str_p)*Decimal(str(math.log2(Decimal(str_p))))
        h+=p*math.log2(p)
    return -h

def get_entropy_by_data(data_list,feat_idx):
    h=0.0
    # 获取特征的取值列表
    feat_set=set()
    y_set=set()
    # k=特征取值 v=count
    feat_dict=dict()
    y_dict=dict()
    for row in data_list:
        feat_set.add(row[feat_idx])
        y_set.add(row[-1])
        y_key='-1_'+row[-1]
        set_feat_count(y_dict,y_key)
        key=feat_idx+'_'+row[feat_idx]
        set_feat_count(feat_dict, key)
    # 总样本数
    total=len(data_list)
    y_h=0.0
    #计算y的信息熵
    for y in y_set:
        p=y_dict['-1_'+y]/total
        y_h-=p*math.log2(p)
    if feat_idx==-1:
        h=y_h
    else:
        for i in feat_set:
            p=feat_dict[feat_idx+'_']

    return h

'''
特征-取值-样本数
'''
def set_feat_count(feat_dict, key):
    if feat_dict.get(key, -1) == -1:
        feat_dict[key] = 0
    feat_dict[key] += 1.0


# print(math.log2(Decimal('2.0')))

# ps=[128.0/384,256.0/384]
# print(get_entropy(ps))
# ps=[1,0]
# print(get_entropy(ps))
# ps=[256.0/384,128.0/384]
# print(get_entropy(ps))
#
# ps=[0.375,0.625]
# print(get_entropy(ps))

'''
计算条件熵 H(D|A)=sum_i(Di/D * H(Di))
Di:特征A取值为i的样本数
H(Di)=sum_k(D_ik/D_i * log(D_ik/D_i))
'''
def get_condition_entropy(cond_datas):
    res=0.0
    for i_list in cond_datas:
        res+=i_list[0]*get_entropy(i_list[1])

    return res

cond_datas=[
            [384.0/1024,[128.0/384,256.0/384]],
            [256/1024,[1,0]],
            [384/1024,[256.0/384,128.0/384]]
            ]
# print(get_condition_entropy(cond_datas))

'''
获取信息增益最大的特征
'''
# def get_best_feat(data_list,cols):
#     best_feat_entropy=0.0
#     best_feat_idx=0
#     # 遍历特征 计算每个特征的条件信息熵
#     for feat in cols:
#
#
#
#     return 1

'''
根据list获取众数
'''
def get_most_y(class_list):
    if(len(class_list)==0):
        return -1
    res=dict()

    for y in class_list:
        if res.get(y,-1)==-1:
            res[y]=0
        res[y]+=1
    res=sorted(res.items(),key=lambda item:item[1],reverse=True)
    return res[0][0]


def create_tree(data_list,cols):
    if(len(data_list)==0):
        return -1
    class_list=[row[-1] for row in data_list]
    # 判断数据是否纯净 如：所有y=1
    if(len(set(class_list))==1):
        return class_list[0]
    # 判断列表每行数据是否只有一维特征 即只剩y，则返回y中取值较多的那个值 [1,1,1,2,2]=>1
    if(len(data_list[0])==1):
        return get_most_y(class_list)
    # 获取增益最大的特征 进行建树

    # 遍历该特征的所有取值
        #获取该值对应的样本 进行递归建树
    return 1

cl_list=[1,1,2,3]
print(get_most_y(cl_list))