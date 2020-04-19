import jieba

dict_file = 'user_dict.txt'
jieba.load_userdict(dict_file)

s = '中国好声音是一个中国综艺节目'
s1 = '中文分词可以在大数据处理中应用，当然也可以在深度学习中使用'

# print(' '.join(jieba.cut(s,cut_all=True)))
print(' '.join(jieba.cut(s1,cut_all=True)))