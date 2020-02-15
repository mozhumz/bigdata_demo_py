import os
import math

file_path = '../data/allfiles'
# 读取停用词表
stop_set = set()
with open('stop_word.txt', 'r',encoding='utf-8') as f:
    for word in f.readlines():
        stop_set.add(word.strip())

doc_words = dict()
doc_num = 0
for filename in os.listdir(file_path): # ls 当前目录下的所有文章名字
    # print(filename)
    with open(file_path+'/'+filename,'r',encoding='utf-8') as f:
        # print(f.read())
        word_freq = dict()
        sum_cnt = 0  # 统计占比用
        max_tf = 0  # 使用最大词频的单词处理
        for line in f.readlines():
            words = line.strip().split(' ')
            for word in words:
                if len(word.strip()) < 1 or word in stop_set:
                    continue
                if word_freq.get(word, -1) == -1:
                    word_freq[word] = 0
                word_freq[word] += 1
                sum_cnt += 1
                if word_freq[word] > max_tf:
                    max_tf = word_freq[word]
        # print(word_freq)
        # print('\n')

        # 将词频处理成占比形式
        for word in word_freq.keys():
            # word_freq[word] /= sum_cnt
            word_freq[word] /= max_tf
        # print(word_freq)

        doc_words[filename] = word_freq
        doc_num += 1
    # print(doc_words)

# 统计没个词的doc-freq（df）
doc_freq = dict()
for doc in doc_words.keys():  # 文本名字
    for word in doc_words[doc].keys():
        if doc_freq.get(word, -1) == -1:
            doc_freq[word] = 0
        doc_freq[word] += 1
print(doc_num)
# print(doc_freq)

# 套idf公式
for word in doc_freq.keys():
    doc_freq[word] = math.log(doc_num/float(doc_freq[word]+1),10)
# print(doc_freq)

# print(sorted(doc_freq.items(),key=lambda x:x[1],reverse=True)[:10])
# print(sorted(doc_freq.items(),key=lambda x:x[1],reverse=False)[:10])


# 套公式tf*idf

for doc in doc_words.keys():
    for word in doc_words[doc].keys():
        doc_words[doc][word] *= doc_freq[word]

print(sorted(doc_words['3business.seg.cln.txt'].items(),key=lambda x:x[1],reverse=True)[:10])
print(sorted(doc_words['3business.seg.cln.txt'].items(),key=lambda x:x[1],reverse=False)[:10])