import os
import sys
import random

file_path = '../data/allfiles'

# 训练集和测试集输出路径
TrainOutFilePath = '../data/mid_data/data.train'
TestOutFilePath = '../data/mid_data/data.test'
# 划分数据集train，test 0.8，0.2
TrainingPercent = 0.8

train_out_file = open(TrainOutFilePath,'w',encoding='utf-8')
test_out_file = open(TestOutFilePath,'w',encoding='utf-8')

wordIDDict = dict()
label_dict = {'business':0,'yule':1,'it':2,'sports':3,'auto':4}

tag = 0
for filename in os.listdir(file_path):
    if filename.find('business') != -1:
        tag = label_dict['business']  # 0
    elif filename.find('yule') != -1:
        tag = label_dict['yule']
    elif filename.find('it') != -1:
        tag = label_dict['it']
    elif filename.find('sports') != -1:
        tag = label_dict['sports']
    else:
        tag = label_dict['auto']

    rd = random.random() #[0-1]
    outfile = test_out_file
    if rd< TrainingPercent: # 0.8
        outfile = train_out_file

    outfile.write(str(tag)+' ')

    with open(os.path.join(file_path,filename),'r',encoding='utf-8') as f:
        content = f.read().strip()
        words = content.replace('\n',' ').split(' ')
        for word in words:
            if len(word.strip())<1 : # 可以在判断中加停用词表
                continue
            # 词不在字典中，将词编码
            if wordIDDict.get(word,-1) == -1:
                wordIDDict[word] = len(wordIDDict)
            outfile.write(str(wordIDDict[word])+' ')
    outfile.write('#' + filename+'\n')

print(len(wordIDDict),'unique words found!')

