import os
import math
import random
import operator
# from k_means import load_data
# file_path = './data'  # 数据路径
# K=3
# doc_dict,doc_list = load_data()
# doc1_dict=doc_dict['734business']
# doc2_dict = doc_dict['1643auto']
# words = set(doc1_dict).union(set(doc2_dict))
# print(len(words))
# print(words)
# k_doc_list = random.sample(doc_list, K)
# print(doc_list)
# print(k_doc_list)
# print(random.random()*27)
# import re
# label_pattern = re.compile(r'[a-z]+')
# print(label_pattern.findall('12business'))

from collections import Counter
a = dict()
for i in range(3):
    a[i] = "%s"%i
print([i for i in a.keys()])
print(Counter([i for i in a.keys()]))
print(abs(-1))