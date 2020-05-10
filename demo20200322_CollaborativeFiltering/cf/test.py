import jieba
import cpca
a = ['a', 'b', 'c']
b = ['a', 'a', 'd']

# print(set(b))  # {'a', 'd'}
# print(set(a) & set(b))  # {'a'}
# print(len(set(a) & set(b)))  # 1

C=dict()
C[1]=dict()
C[1][2]='a'
C[1][3]='b'
C[1][4]='c'

# for u,sim_users in C.items():
#     print(u)
#     print(sim_users)
#     for v,cuv in sim_users.items():
#         print(v)
#         print(cuv)


str="成都市的武侯区发生了火灾，好像是在洗面桥街"
res=' '.join(jieba.cut(str,cut_all=True))
print(res)

location_str = ["徐汇区虹漕路461号58号楼5楼", "泉州市洛江区万安塘西工业区", "朝阳区北苑华贸城"]
df = cpca.transform(location_str)

df2=cpca.transform(["成都的武侯区发生了火灾，好像是在洗面桥街"], cut=False)

print(df2)