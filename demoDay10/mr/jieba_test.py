import jieba

# a = '我今天要上Map Reduce，我今天上午已经上了课。'
# cut_lst = [x for x in jieba.cut(a,cut_all=False)]
# print(cut_lst)
# print([hash(x) % 3 for x in cut_lst])




a1 = '我'
a2 = '今天'
a3 = '要'
a4 = '我'
#
print(a1,hash(a1),hash(a1)%3)
print(a2,hash(a2),hash(a2)%3)
print(a3,hash(a3),hash(a3)%3)
print(a4,hash(a4),hash(a4)%3)