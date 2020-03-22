a = ['a', 'b', 'c']
b = ['a', 'a', 'd']

print(set(b))  # {'a', 'd'}
print(set(a) & set(b))  # {'a'}
print(len(set(a) & set(b)))  # 1

C=dict()
C[1]=dict()
C[1][2]='a'
C[1][3]='b'
C[1][4]='c'

for u,sim_users in C.items():
    print(u)
    print(sim_users)
    for v,cuv in sim_users.items():
        print(v)
        print(cuv)