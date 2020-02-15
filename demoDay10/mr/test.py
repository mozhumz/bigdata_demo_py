# a = 'a b c'
# # print(a.strip())
# print(a)
# # print(a.strip())
# print(a.split(' '))

import re
word = '--?'  # note2
p = re.compile(r'\w+')
word = p.findall(word)   # ['grandfather']
print(word)
print(word[0])
print(word[0].lower())

# # list index out of range数组越界
# a = [1,2,3,4]
# print(a[4])