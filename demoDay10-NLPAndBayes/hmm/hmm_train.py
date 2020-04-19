import math

data_path = '../data/allfiles.txt'
mode_path = '../data/mid_data/hmm.mod'
# 一、初始化模型参数（π初始状态，a转移矩阵，b发射矩阵）
# 其中状态为：B,M,E,S（每个中文字都会对应其中的一个状态）
STATUS_MUN = 4  # M=4

# 1. 初始状态概率 π    [0.0, 0.0, 0.0, 0.0]
pi = [0.0 for pi in range(STATUS_MUN)]   # count
pi_sum = 0.0
# print(pi)

# 2.状态转移概率 a：M*M矩阵
# A:
# [0.0, 0.0, 0.0, 0.0] 分母1
# [0.0, 0.0, 0.0, 0.0] 分母2
# [0.0, 0.0, 0.0, 0.0] 分母3
# [0.0, 0.0, 0.0, 0.0] 分母4
A = [[0.0 for col in range(STATUS_MUN)] for row in range(STATUS_MUN)]  # count
A_sum = [0.0 for row in range(STATUS_MUN)] # 存储每一行数据的分母，用作求概率

# 3. 发射概率 b： M*N矩阵 本身矩阵相对比较稀疏，因为是4个状态到对应每个中文字的概率
# N是中文字典大小，每个字不一定都会有4个状态，或者有些字在对应状态中没有出现过导致的数据稀疏
# 所以这个不需要矩阵，用字典稀疏存储减少空间浪费
B = [dict() for row in range(STATUS_MUN)]
B_sum = [0.0 for row in range(STATUS_MUN)]


f_txt = open(data_path,'r',encoding='utf-8')

# 将词转化成单个字的列表
def get_word_ch(word):
    ch_lst = []
    for ch in word:
        ch_lst.append(ch)
    return ch_lst

# print(get_word_ch("动态规划"))

while True:
    line = f_txt.readline() # 读一行相当于一篇文章
    # 读完所有文章推出循环
    if not line:
        break

    words = line.strip().split()

    ch_lst = []   # 每个词所对应的中文单个字的数组 （动态规划）['动'，'态'，'规'，'划' ]
    status_lst = []  # 对应单个中文字的状态数组 [BMES]=>[0,1,2,3]
    for word in words[:-1]:
        cur_ch_lst = get_word_ch(word)
        cur_ch_num = len(cur_ch_lst)  # 这个词有多少个字

        # 初始化字符状态
        cur_stauts_lst = [0 for ch in range(cur_ch_num)]
        # S:3
        if cur_ch_num == 1:
            cur_stauts_lst[0] = 3
        else: # 否则就是BME
            # 标识B：0
            # cur_stauts_lst[0] = 0 # 因为初始化已经标识为0了
            # 标识E：2
            cur_stauts_lst[-1] = 2
            # 中间的全部为M：1
            for i in range(1,cur_ch_num-1):
                cur_stauts_lst[i] = 1
        ch_lst.extend(cur_ch_lst)
        status_lst.extend(cur_stauts_lst)
    # ch_lst,status_lst 每篇文章的字和状态

    # 做模型count统计（分子部分） sum部分（分母部分）
    for i in range(len(ch_lst)):  # 扫一遍文字序列，对状态序列进行统计
        cur_status = status_lst[i]  # 获取当前文字的状态
        cur_ch = ch_lst[i]
        # 统计初始概率π
        if i == 0:
            # 为什么不用BMES用0123，就是方便状态直接可以作为索引形式
            pi[cur_status] += 1.0     # 状态分子部分相加
            pi_sum += 1.0  # 不管是哪个状态，分母都+1，sum，为了求概率
        # 统计发射概率B
        if B[cur_status].get(cur_ch, -1) == -1:
            B[cur_status][cur_ch] = 0.0
        B[cur_status][cur_ch] += 1.0
        B_sum[cur_status] += 1.0

        # 状态转移概率 A
        if i+1 < len(ch_lst):
            A[cur_status][status_lst[i+1]] += 1.0
            A_sum[cur_status] += 1.0

f_txt.close()

# 将统计结果转化成概率形式
for i in range(STATUS_MUN):
    # pi
    # pi[i] = pi[i]/pi_sum
    # pi[i] /= pi_sum
    # 因为pi[i]等于0，去不了log，属于负无穷
    pi[i] = -100000.0 if pi[i] == 0.0 else math.log(pi[i]/pi_sum)

    # A
    for j in range(STATUS_MUN):
        A[i][j] = -10000.0 if A[i][j] == 0.0 else math.log(A[i][j]/A_sum[i])
    # B
    for ch in B[i]:
        B[i][ch] = math.log(B[i][ch]/B_sum[i])

# 存储模型->模型文件
f_mod = open(mode_path,'w',encoding='utf-8')
f_mod.write(str(pi)+'\n')
f_mod.write(str(A)+'\n')
f_mod.write(str(B)+'\n')

f_mod.close()







