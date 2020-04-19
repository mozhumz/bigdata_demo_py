mode_path = '../data/mid_data/hmm.mod'
filter_val = 0.02
# 1.加载模型
with open(mode_path,'r',encoding='utf-8') as f:
    # 初始状态概率
    pi = eval(f.readline())
    # 状态转移概率
    A = eval(f.readline())
    # 发射概率
    B = eval(f.readline())
# 2. 实现viterbi算法（预测，获取最优路径，得到概率最大的状态组合方式）

STATUS_NUM = 4


# 将词转化成单个字的列表
def get_word_ch(word):
    ch_lst = []
    for ch in word:
        ch_lst.append(ch)
    return ch_lst


def viterbi(sentence,sep=' '):
    ch_lst = get_word_ch(sentence)
    # print(ch_lst)
    # 文本序列的长度T
    ch_num = len(ch_lst)

    # 初始化状态矩阵
    # 三维数组 最外层为行状态M，每行是单字符串数组，
    # 数组第一个元素为发射概率*初始状态概率 第二个元素为状态的下标
    status_matrix = [[[0.0,0] for col in range(ch_num)] for row in range(STATUS_NUM)]

    # 状态种类M个  m*1
    for i in range(STATUS_NUM):
        # 第一个字的发射概率
        if B[i].get(ch_lst[0],-1) != -1:
            cur_B = B[i][ch_lst[0]]
        else:
            cur_B = -100000.0
        #初始状态概率
        cur_pi = pi[i]
        status_matrix[i][0][0] = cur_pi + cur_B  # 取了log相乘就是相加
        status_matrix[i][0][1] = i

    # M*M*(T-1)
    for i in range(1,ch_num): # 1:T  (T-1)
        for j in range(STATUS_NUM):  # M
            max_p = None
            max_status = None
            # 经过j节点的前M个节点中存储概率最大值，及上一层过来到这个节点的编号
            for k in range(STATUS_NUM): # M 前一层的第k个s
                # k到j的转移概率
                cur_A = A[k][j]
                cur_p = status_matrix[k][i-1][0]+cur_A
                if max_p is None or max_p < cur_p:
                    max_p = cur_p
                    max_status = k

            # 当前字符的发射概率
            if B[j].get(ch_lst[i],-1)!=-1:
                cur_B = B[j][ch_lst[i]]
            else:
                cur_B = -100000.0
            status_matrix[j][i][0] = max_p+cur_B
            status_matrix[j][i][1] = max_status

    # get max prob path
    last_prob_lst = []

    # 找最后一层中概率最大的，也就是最终路径最大的
    # for i in range(STATUS_NUM):
    #     if max_end_p is None or status_matrix[i][ch_num-1][0] > max_end_p:
    #         max_end_p = status_matrix[i][ch_num-1][0]
    #         max_end_status = i
    for i in range(STATUS_NUM):
        last_prob_lst.append((status_matrix[i][ch_num-1][0],status_matrix[i][ch_num-1][1],i))
    sorted_last_lst = sorted(last_prob_lst, key=lambda x:x[0], reverse=True)
    print(sorted_last_lst)


    def get_best_seg(max_end_status):
        # 根据之前记录的往回找最优路径
        best_status_lst = [0 for ch in range(ch_num)]
        best_status_lst[ch_num - 1] = max_end_status

        c = ch_num-1
        cur_best_status = max_end_status
        while c > 0:
            # 上一个最好的状态
            pre_best_status = status_matrix[cur_best_status][c][1]
            best_status_lst[c-1] = pre_best_status
            cur_best_status = pre_best_status
            c -= 1

        # 实现切词
        rest = ''
        rest +=ch_lst[0]
        for i in range(1,ch_num):
            # i-1 是E,S 或者 i是B,S
            if best_status_lst[i-1] in {2,3} or best_status_lst[i] in {0,3}:
                rest += sep
            rest += ch_lst[i]
        return rest

    i = 0
    res = []
    max_end_p = sorted_last_lst[0][0]
    while (max_end_p - sorted_last_lst[i][0]) < filter_val:
        # 调用获取最优切分方法
        max_end_status = sorted_last_lst[i][2]
        res.append(get_best_seg(max_end_status))
        i += 1
    return res


if __name__ == '__main__':
    s = '得到概率最大的状态组合方式'
    s1 = '结巴分词里会结合词性，状态会更多'
    s2='k到j的转移概率'
    print(viterbi(s,' '))