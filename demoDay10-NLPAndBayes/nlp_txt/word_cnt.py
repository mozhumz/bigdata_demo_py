import jieba
s1 = "这只皮靴号码大了。那只号码合适"
s2 = "这只皮靴号码不小，那只更合适"
# ##### 1 #####
# s1_seg = '/'.join([x for x in jieba.cut(s1,cut_all=True) if x!=''])
s1_seg = [x for x in jieba.cut(s1,cut_all=True) if x!='']
s2_seg = [x for x in jieba.cut(s2,cut_all=True) if x!='']
#
# ###### 2 ######
s1_set = set(s1_seg)
s2_set = set(s2_seg)
s_set = s1_set.union(s2_set)
#
#
print('s1 seg:',s1_seg)
# print('s1 set: ', s1_set)
# print('\n')
print('s2 seg:',s2_seg)
# print('s2 set: ', s2_set)
# print('\n')
# print(s_set)
#
# #### 3 #####
#
# def word_cnt(s1_seg):
#     s1_dict = {}
#     for word in s1_seg:
#         if s1_dict.get(word)==None:
#             s1_dict[word] = 0
#         s1_dict[word] += 1
#     return s1_dict
#
# print('s1_dict: ',word_cnt(s1_seg))
# print('s2_dict: ',word_cnt(s2_seg))
#
# ##### 4 #######
# 读取停用词表
stop_set = set()
with open('stop_word.txt','r',encoding='utf-8') as f:
    for word in f.readlines():
        stop_set.add(word.strip())
        # print(stop_set)
s1_seg_new = [x for x in s1_seg if x not in stop_set]
print(s1_seg_new)
# 1)对字典中的词进行编码
# index = 0
word_encode_dict = {}
for word in s_set:
    if word in stop_set:
        continue
    word_encode_dict[word] = len(word_encode_dict)
# print('word encode :',word_encode_dict)
#
# # 2)
# s1_wc = word_cnt(s1_seg)
# s2_wc = word_cnt(s2_seg)
#
# s1_vec = [0]*len(s_set)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# s2_vec = [0]*len(s_set)
# print('s1 vec: ',s1_vec)
# print('s2 vec: ',s2_vec)
# print('\n')
# # print(s1_wc.items())  # dict_items([('皮靴', 1), ('这', 1), ('只', 2), ('号码', 2), ('那', 1), ('合适', 1), ('大', 1), ('了', 1)])
#
# for w, c in s1_wc.items():
#     s1_vec[word_encode_dict[w]] = c
#
# for w, c in s2_wc.items():
#     s2_vec[word_encode_dict[w]] = c
#
# print('s1 vec: ',s1_vec)
# print('s2 vec: ',s2_vec)

def word_vec(s1,s2):
    s1_seg = [x for x in jieba.cut(s1, cut_all=True) if x != '']
    s2_seg = [x for x in jieba.cut(s2, cut_all=True) if x != '']

    s_set = set(s1_seg).union(set(s2_seg))  # set(s1_seg)|set(s2_seg)

    def word_cnt(s1_seg):
        s1_dict = {}
        for word in s1_seg:
            if s1_dict.get(word) == None:
                s1_dict[word] = 0
            s1_dict[word] += 1
        return s1_dict

    # 加载停用词表
    stop_set = set()
    with open('stop_word.txt', 'r', encoding='utf-8') as f:
        for word in f.readlines():
            stop_set.add(word.strip())
    word_encode_dict = {}
    for word in s_set:
        if word in stop_set:
            continue
        word_encode_dict[word] = len(word_encode_dict)
    s1_wc = word_cnt(s1_seg)
    s2_wc = word_cnt(s2_seg)

    s1_vec = [0] * len(word_encode_dict)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    s2_vec = [0] * len(word_encode_dict)

    for w, c in s1_wc.items():
        if word_encode_dict.get(w)==None:
            continue
        s1_vec[word_encode_dict[w]] = c

    for w, c in s2_wc.items():
        if word_encode_dict.get(w)==None:
            continue
        s2_vec[word_encode_dict[w]] = c
    return s1_vec,s2_vec


s1_vec,s2_vec = word_vec(s1,s2)
print('s1 vec: ',s1_vec)
print('s2 vec: ',s2_vec)

import math
sum_dot = 0
sum_s1 = 0.0
sum_s2 = 0.0
print([x for x in zip(s1_vec,s2_vec)])
for x,y in zip(s1_vec,s2_vec):
    sum_dot += x*y
    sum_s1 += x*x
    sum_s2 += y*y
cos_sim = sum_dot/(math.sqrt(sum_s1)*math.sqrt(sum_s2))
print(cos_sim)