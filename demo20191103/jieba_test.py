# coding=utf-8
import jieba

a = '我今天要上Map Reduce，我今天上午已经上了课。'
cut_lst = [x for x in jieba.cut(a,cut_all=False)]
print(cut_lst)
for w in cut_lst:
    print w