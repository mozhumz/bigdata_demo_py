
import re
p = re.compile(r'\w+')
# data_path = './data/test.txt'
data_path = '../data/The_man_of_property.txt'
with open(data_path,'r', encoding='utf-8') as f:
    # for i in range(2):
    # print(f.readlines())
    for line in f.readlines():  # ['a b c\n', 'c b d']
        word_lst = line.strip().split(" ")  # ['a', 'b', 'c']
        # print(word_lst)
        for word in word_lst:
            re_word = p.findall(word)
            if len(re_word)==0:
                continue
            word = re_word[0].lower()
            print('%s,%s'%(word,1))