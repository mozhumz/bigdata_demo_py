import os
import math
import random
import operator
# from k_means import load_data
# file_path = './data'  # 数据路径
K=3
# doc_dict,doc_list = load_data()
# doc1_dict=doc_dict['734business']
# doc2_dict = doc_dict['1643auto']
# words = set(doc1_dict).union(set(doc2_dict))
# print(len(words))
# print(words)
doc_list=[1,1,2,2,3,4,5]
k_doc_list = random.sample(doc_list, K)
print(doc_list)
print(k_doc_list)
# print(random.random()*27)
# import re
# label_pattern = re.compile(r'[a-z]+')
# print(label_pattern.findall('12business'))

# from collections import Counter
# a = dict()
# for i in range(3):
#     a[i] = "%s"%i
# print([i for i in a.keys()])
# print(Counter([i for i in a.keys()]))
# print(abs(-1))
new_WCSS=343922.3069810851
WCSS=326257.63683079596
WCSS_sub = abs(new_WCSS - WCSS)
# print(WCSS_sub)

a=1
b=a
a=0
print(b)
wcss_list_d=dict()
wcss_list_d[1]=1
wcss_list_d[2]=1
print(wcss_list_d)
k_w=sorted(wcss_list_d.items(),key=lambda x:x[1])[0]
d=dict()
d[1]=k_w
d[2]=[2,3]
# print(sorted(d.items(),key=lambda x:x[1][1])) (7, (2, 328038.93267441314))
res=[(8, (5, 326057.8104891359)), (6, (1, 327321.8444780574)), (7, (2, 328038.93267441314)), (3, (3, 329621.4846246948)), (4, (1, 329815.8961249155)), (5, (3, 335531.77469469566)), (2, (0, 340537.9757117981)), (1, (0, 341503.9192205848))]
res.sort(key=lambda x:x[0])

# key=k v=[(k,(ite_num,wcss),sub]
wcss_d=dict()
sub=1
wcss=0
for x in res:
    new_wcss=x[1][1]
    sub=abs(new_wcss-wcss)
    wcss=new_wcss
    wcss_d[x[0]]=[x,sub]
    print(x)
    print(sub)
    print('-----------------------')

d1=sorted(wcss_d.items(),key=lambda x:(x[1][1],x[0]))

d2=d1[:4]
d2.sort(key=lambda x:x[0])
print(d2)

import re
label_pattern = re.compile(r'[a-z]+')
res=label_pattern.findall('adb12hy44tt 45dd 99oo rrd tt')
print(res)
K=3
doc_dict={'1':{'1':1},'2':{'2':2},'3':{3:3},4:{4:4}}
doc_dict2={'1':{'1':1},'2':{'2':2},'3':{3:3},45:{4:4}}
union_set=set(doc_dict)|set(doc_dict2)
print(union_set)
# rs=random.sample(doc_dict.keys(),K)
# # print(rs)
# for item in doc_dict.items():
#     print(item[1])


