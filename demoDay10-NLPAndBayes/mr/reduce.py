
data_path = '../data/reduce_test'
cur_word = None  # null
sum = 0
with open(data_path,'r', encoding='utf-8') as f:
    for line in f.readlines():
        word,val = line.strip().split(',')
        if cur_word == None:
            cur_word = word
        if cur_word!=word:
            print('%s,%s'%(cur_word,sum))
            cur_word=word
            sum = 0
        sum += int(val)  # sum = sum+val
    print('%s,%s' % (cur_word, sum))
