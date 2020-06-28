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

print(math.log2(Decimal('2.0')))

ps=[128.0/384,256.0/384]
print(get_entropy(ps))
ps=[1,0]
print(get_entropy(ps))
ps=[256.0/384,128.0/384]
print(get_entropy(ps))

ps=[0.375,0.625]
print(get_entropy(ps))

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