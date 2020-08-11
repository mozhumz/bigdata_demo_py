import pandas as pd
import math
import numpy as np
df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.],
                   'Col-c':['c1','c2','c3','c4']
                   })
#
# # print(df)
#
# df_g=df.groupby(['Animal']).agg(list).rename(columns={'Max Speed':'Max Speed-arr','Col-c':'Col-c-arr'})
# print(df_g)
#
# a=[1,2]
# b=[1,2]
# # 引用比较
# print(a is b)
# # 内容比较
# print(a==b)
#
# def getHAD(a_list,total):
#     res=.0
#     for i in a_list:
#         res-=i/total*math.log2(i/total)
#     return res
# total=100.
# a_list=[50,50]
# print(getHAD(a_list,total))
# print('-------------------')
# a_list=[50,40,10]
# print(getHAD(a_list,total))
# order_id='1'
# labels=[]
# user_products=['1','2','3']
# train=['2','3']
# labels += [(order_id, product) in train for product in user_products]
# print(labels)
arr1=np.array(["1","2","3"],dtype=np.str)
# print(arr1)
# print('abc'+(arr1[0]))

print(df)
df.set_index('Col-c',drop=False,inplace=True)
print(df[0])