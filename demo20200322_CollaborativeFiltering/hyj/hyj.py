u={1:5,2:3,3:1}
v={1:5,2:3,3:1,7:1,8:9}

# print(set(u.keys())&set(v.keys()))

C={'a':2,'e':3,'f':8,'d':4}
list=sorted(C.items(),key=lambda x:x[1],reverse=True)[:3]
print(list)
# [('f', 8), ('e', 3), ('d', 4)]

# [('f', 8), ('d', 4), ('e', 3)]