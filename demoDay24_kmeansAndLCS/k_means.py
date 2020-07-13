import os
import math
import random
import operator
import re
from collections import Counter

# K = 10  # 设定类别数量（簇）
WCSS = 0.0  # 初始化wcss
new_WCSS = 1  # 初始化
threshold = 1e-6  # 认为不变动的阈值 0.000001
ITER_MAX = 30  # 设定最大迭代次数

file_path = 'F://八斗学院//视频//14期正式课//00-data//data'  # 数据路径

label_dict = {'business': 0, 'yule': 1, 'it': 2, 'sports': 3, 'auto': 4}

word_dict = dict()  # 对word进行编码


def load_data():
    '''
    加载新闻数据，并做词频统计（word count）
    :return: doc_dict{文档名：word_freq}, doc_label {文档名：label}
    '''
    label_pattern = re.compile(r'[a-z]+')
    doc_label = dict()
    doc_dict = dict()
    i = 0

    for filename in os.listdir(file_path):  # filename:1business
        doc_name = filename.split('.')[0]
        label = label_pattern.findall(doc_name)[0]
        doc_label[doc_name] = label
        if i % 100 == 0:
            print(i, 'files loaded!!')
        with open(file_path + '/' + filename, 'r', encoding='utf-8') as f:
            word_freq = dict()  # tf，统计逻辑word count，结果需要的字典结构
            for line in f.readlines():
                words = line.strip().split(' ')
                for word in words:
                    if len(word.strip()) < 1:
                        continue
                    # 对word进行编码 0,1,2,3,4,5
                    if word_dict.get(word, -1) == -1:
                        word_dict[word] = len(word_dict)
                    wid = word_dict.get(word, -1)
                    # word count统计逻辑
                    if word_freq.get(wid, -1) == -1:
                        word_freq[wid] = 1
                    else:
                        word_freq[wid] += 1

            doc_dict[doc_name] = word_freq
        i += 1
    return doc_dict, doc_label


def idf(doc_dict):
    '''
    统计每个单词的文档频率
    :param doc_dict: {文档名：word_freq}
    :return: word_idf {word：idf值}
    '''
    word_idf = {}
    # 统计doc freq  doc_dict {word：出现的doc数量}
    for doc in doc_dict.keys():  # doc_dict{文档名：word_freq}
        for word in doc_dict[doc].keys():
            if word_idf.get(word, -1) == -1:
                word_idf[word] = 1
            else:
                word_idf[word] += 1
    doc_num = len(doc_dict)
    # 计算idf
    for word in word_idf.keys():
        word_idf[word] = math.log(doc_num / (word_idf[word] + 1))
    return word_idf


def doc_tf_idf():
    '''
    实现tf*idf,计算每篇文章中对应每个单词的tf-idf值
    :return: doc_dict {文档名：{单词：tf-idf值}}, doc_list文档名列表
    '''
    doc_dict, doc_label = load_data()
    word_idf = idf(doc_dict=doc_dict)

    for doc in doc_label.keys():
        for word in doc_dict[doc].keys():
            doc_dict[doc][word] = doc_dict[doc][word] * word_idf[word]  # tf*idf
    return doc_dict, doc_label


def init_K(doc_dict, doc_list,K=3):
    '''
    初始化K个中心点，随机选择样本点为中心点
    :param doc_dict: 样本数据，每个doc是一条样本
    :param doc_list: 样本数据doc名
    :return:
    '''
    print('K:'+str(K))
    center_dict = dict()
    k_doc_list = random.sample(doc_list, K)
    i = 0
    for doc_name in k_doc_list:
        center_dict[i] = doc_dict[doc_name]   # dense [0.3,0.2]  sparse:dict 0:0.3,1:0.2...
        i += 1
    return center_dict


def compute_dis(doc1_dict, doc2_dict):
    '''
    计算样本与样本之间的距离
    :param doc1_dict: 样本数据 {word:tf*idf}
    :param doc2_dict: 另一个样本数据 {word:tf*idf}
    :return: sum两个样本的欧式距离
    '''
    sum = 0.0
    # 两个文档总共的去重单词数
    words = set(doc1_dict).union(set(doc2_dict))
    for wid in words:
        d = doc1_dict.get(wid, 0.0) - doc2_dict.get(wid, 0.0)
        sum += d * d
    return math.sqrt(sum)


def compute_center(doc_list, doc_dict):
    '''
    重新计算其中一个样本点的中心点
    :param doc_list: 属于第k个类别的所有样本
    :param doc_dict: 样本字典 {文档名：{单词wid：tf-idf值}}
    :return: 中心点（坐标），因为维度比较多存储到dict中
    '''
    tmp_center = dict()

    for doc in doc_list:
        for wid in doc_dict[doc].keys():
            if tmp_center.get(wid, -1) == -1:
                tmp_center[wid] = doc_dict[doc][wid]
            else:
                tmp_center[wid] += doc_dict[doc][wid]

    for wid in tmp_center.keys():
        tmp_center[wid] /= len(doc_list)
    return tmp_center


def all_k_dist(doc_list, doc_dict, k_dict):
    '''
    计算所有的样本与对应中心点之间的距离
    :param doc_label:
    :param doc_dict:
    :param k_dict:
    :return:
    '''
    sum = 0.0
    for doc in doc_list:
        sum += compute_dis(doc_dict[doc], k_dict)
    return sum

'''
# key=k+迭代次数 v=wscc
'''
def do_kmeans(wcss_d,k_count):
    print('k_count'+str(k_count))
    global WCSS, new_WCSS
    # 初始化K个中心点到样本上
    center_dict = init_K(doc_dict, doc_label.keys(),k_count)
    # 初始化doc所属类别，在k=0的中心点上
    doc_k = dict(zip(doc_label.keys(), [0 for i in range(len(doc_label.keys()))]))
    iter_num = 0
    Center_mv = 1
    k_doc = dict()
    WCSS_sub = 1
    print('start train!!')
    wcss_list_d=dict()
    # 算法循环终止条件,只要有其中一个条件不满足跳出循环
    while WCSS_sub > threshold and iter_num < ITER_MAX and Center_mv > threshold:
        tmp_loss=0
        k_doc = dict()
        # 对每一个样本算到所有k中心点的距离 ，样本点归属离哪个中心点最近归属哪个中心点的类别
        for doc in doc_label.keys():
            tmp_select_k = dict()  # 每个样本到k个类别的距离
            for k in center_dict.keys():
                tmp_select_k[k] = compute_dis(doc_dict[doc], center_dict[k])
            # (k, val) = sorted(tmp_select_k.items(), key=operator.itemgetter(1))[0]
            (k, val) = min(tmp_select_k.items(), key=operator.itemgetter(1))
            tmp_loss+=val
            doc_k[doc] = k
            if k_doc.get(k, -1) == -1:
                k_doc[k] = [doc]
            else:
                k_doc[k].append(doc)

        # step 2: 重新计算中心点
        Center_mv = 0
        # WCSS赋值
        WCSS = 0 if new_WCSS == 0 else new_WCSS
        new_WCSS = 0
        for k in k_doc.keys():
            # 重新计算中心点
            tmp_k_center = compute_center(k_doc[k], doc_dict)
            # 计算所有中心点移动的距离
            Center_mv += compute_dis(center_dict[k], tmp_k_center)
            # 计算wcss公式
            new_WCSS += all_k_dist(k_doc[k], doc_dict, tmp_k_center)
            # 更新k中心点坐标
            center_dict[k] = tmp_k_center
        WCSS_sub = abs(new_WCSS - WCSS)
        # 存储迭代次数和wcss
        wcss_list_d[iter_num]=new_WCSS
        print(iter_num, WCSS_sub, str(Center_mv) + "\t WCSS: " + str(new_WCSS),'tmp_loss:'+str(tmp_loss))
        iter_num += 1

    # 存储每次迭代的Wcss
    k_w=sorted(wcss_list_d.items(),key=lambda x:x[1])[0]
    wcss_d[k_count]=k_w
    # write doc-> cluster
    for k in k_doc.keys():
        cluster_doc_wc = Counter([doc_label.get(doc_name) for doc_name in k_doc[k]])
        print(sorted(cluster_doc_wc.items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    # 读取数据，获得每篇文章中单词的tf-idf值
    doc_dict, doc_label = doc_tf_idf()
    f1='1business'
    f2='1yule'
    print(compute_dis(doc_dict[f1],doc_dict[f2]))
    # key=k v=[item_num,wcss]
    wcss_d=dict()
    for i in range(3,9):
        do_kmeans(wcss_d,i)
    print(sorted(wcss_d.items(),key=lambda x:x[1][1]))
