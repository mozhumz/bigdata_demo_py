# a = ['a', 'b', 'c']
# b = ['a', 'a', 'd']
#
# # print(set(b))  # {'a', 'd'}
# # print(set(a) & set(b))  # {'a'}
# # print(len(set(a) & set(b)))  # 1
# print(set(a)|set(b))
import numpy as np

a = np.array([3,4,5,6,100])
print(a.std(),a.mean())

c = [(x-a.mean())/a.std() for x in a ]
print(c)