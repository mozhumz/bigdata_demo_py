x = 'ABCBDAB'
y = 'BDCABA'

n = len(x)
m = len(y)
l = [[0]*(m+1) for i in range(n+1)]
# print([0]*3)  # [0, 0, 0]
# print(l)
for i in range(1, n+1):
    for j in range(1, m+1):
        if x[i-1] == y[j-1]:
            l[i][j] = l[i-1][j-1]+1
        else:
            l[i][j] = max(l[i-1][j], l[i][j-1])
print(l[-1][-1])

for i in range(len(l)):
    print(l[i])