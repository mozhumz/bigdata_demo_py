import random, os, math, operator, re
from collections import Counter

# 文件路径
file_path = 'F://八斗学院//视频//14期正式课//00-data//data/'
# 文章类别
label_dict = {'business': 0, 'yule': 1, 'it': 2, 'sports': 3, 'auto': 4}
# 对word进行编码
word_dict = dict()
label_pattern = re.compile(r'[a-z]+')

'''1 word count'''
# k=doc_name v={'word':count}
doc_dict = dict()
# k=doc_name v=label
label_dict = dict()
for file_name in os.listdir(file_path):
    label = label_pattern.findall(file_name.split('.')[0])[0]
    label_dict[file_name] = label
    with open(file_path + file_name, mode='r', encoding='utf-8')as f:
        for line in f.readlines():
            words = line.split(' ')
            for word in words:
                if word.strip():
                    # 单词编码
                    if word_dict.get(word, -1) == -1:
                        word_dict[word] = len(word_dict)
                    # count
                    if doc_dict.get(file_name, -1) == -1:
                        doc_dict[file_name] = dict()
                    if doc_dict[file_name].get(word_dict[word], -1) == -1:
                        doc_dict[file_name][word_dict[word]] = 0
                    doc_dict[file_name][word_dict[word]] += 1

# print(doc_dict)
'''2 idf'''
# k=word v=idf
word_idf = dict()
doc_num = len(doc_dict)
for doc_name in doc_dict.keys():
    for word in doc_dict[doc_name].keys():
        if word_idf.get(word, -1) == -1:
            word_idf[word] = 0
        word_idf[word] += 1

for word in word_idf.keys():
    word_idf[word] = math.log(doc_num / (word_idf[word] + 1.))

'''3 tf*idf'''
for doc_name in doc_dict.keys():
    for word in doc_dict[doc_name].keys():
        doc_dict[doc_name][word] *= word_idf[word]
    # print(doc_dict[doc_name])
    # break

'''4 随机选取k个样本作为中心点'''
K = 8
# [doc_name:{word:count}]
docs = random.sample(label_dict.keys(), K)
# k=num v={word:count}
cent_list=dict()
k=0
for doc in docs:
    cent_list[k]=doc_dict[doc]
    k+=1


'''5 计算2个样本的距离'''
def compute_dis(doc1: dict, doc2: dict):
    sum=0.
    words=set(doc1)|set(doc2)
    for word in words:
        v1=doc1.get(word,0.)
        v2=doc2.get(word,0.)
        sum+=(v1-v2)*(v1-v2)

    return math.sqrt(sum)

'''计算中心点'''
# param {n:[{word:score}]}
# return {n:{'word': 'score'}}
def comput_cent(cent_dict:dict):
    cent_res=dict()
    # k=word v=score
    for item in cent_dict.items():
        doc_smple_dict=dict()
        for wd in item[1]:

            for word in wd.keys():
                if doc_smple_dict.get(word,-1)==-1:
                    doc_smple_dict[word]=0
                doc_smple_dict[word]+=wd[word]

        for word in doc_smple_dict.keys():
            doc_smple_dict[word]/=len(item[1])

        cent_res[item[0]]=doc_smple_dict

    wcss=0
    for k in cent_dict.keys():
        tmp_cent=cent_res[k]
        for samp in cent_dict[k]:
            wcss+=compute_dis(samp,tmp_cent)
    return cent_res,wcss


# 循环条件
ite_num = 30
i = 0
WCSS = 1
min_WCSS = 0.1
sub = 1
min_sub = 0.1
f1='1business.seg.cln.txt'
f2='1yule.seg.cln.txt'
print(compute_dis(doc_dict[f1],doc_dict[f2]))
# k=num v=[样本]
while i < ite_num and WCSS > min_WCSS and sub > min_sub:
    cent_dict = dict()
    print('i:' + str(i) + ',WCSS:' + str(WCSS) + ',sub:' + str(sub))
    i += 1
    tmp_WCSS = WCSS
    WCSS = 0
    for doc_name in doc_dict.keys():
        tmp_dis = 0
        tmp_sample = {}
        tmp_cent = {}
        j = True
        tmp_select_k = dict()  # 每个样本到k个类别的距离
        for cent in cent_list.items():
            dis = compute_dis(doc_dict[doc_name], cent[1])
            tmp_select_k[cent[0]]=dis
            # if j:
            #     tmp_dis = dis
            #     tmp_sample = doc_dict[doc_name]
            #     tmp_cent = cent
            #     j=False
            # 获取最小距离的样本
            # if dis < tmp_dis:
            #     tmp_dis = dis
            #     tmp_sample = doc_dict[doc_name]
            #     tmp_cent = cent
        (min_k, min_val) = min(tmp_select_k.items(), key=operator.itemgetter(1))
        if cent_dict.get(min_k, -1) == -1:
            cent_dict[min_k] = list()
        cent_dict[min_k].append(doc_dict[doc_name])

        # WCSS += min_val

    #         重新计算中心点
    cent_list,WCSS = comput_cent(cent_dict)
    sub = abs(WCSS - tmp_WCSS)