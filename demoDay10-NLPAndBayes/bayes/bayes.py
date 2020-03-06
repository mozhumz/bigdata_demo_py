import os
import math
# 训练，测试数据路径
TrainOutFilePath = '../data/mid_data/data.train'
TestOutFilePath = '../data/mid_data/data.test'
# 输出模型路径
ModelPath = '../data/rest_data'
class_defaut = os.path.join(ModelPath,'class_defaut.model')
class_prob = os.path.join(ModelPath,'class_prob.model')
class_feat = os.path.join(ModelPath,'class_feat.model')
word_dict = os.path.join(ModelPath,'word_dict.model')


'''
初始化参数
'''

ClassFeatDict = dict()  # xj和yi的矩阵 count  各类别下各单词词频
# ClassFeatProb = dict()  # 概率
ClassFreq = dict()  # p(yi),数组  各类别的文章数
ClassProb = dict()
ClassDefaultProb = dict() # 未出现词的默认概率，每个类别都不一样
DefautFreq = 0.1
WordDic = set()  #单词id集合
doc_num = 0

# 加载数据
def LoadData():
    global doc_num
    with open(TrainOutFilePath, 'r', encoding='utf-8') as f:
        for sline in f.readlines():  # 一行是一篇文章
            pos = sline.find('#')
            if pos > 0:
                sline = sline[:pos].strip()  # tag + words
            words = sline.split(' ')
            # 类别id
            classid = int(words[0])
            # 如果类别不存在，进行初始化
            if ClassFeatDict.get(classid, -1) == -1:
                ClassFeatDict[classid] = dict()
                # ClassFeatProb[classid] = dict()
                ClassFreq[classid] = 0
            ClassFreq[classid] += 1
            words = words[1:]

            for word in words:
                if len(word.strip()) < 1:
                    continue
                wid = int(word)
                WordDic.add(wid)
                if ClassFeatDict[classid].get(wid, -1) == -1:
                    ClassFeatDict[classid][wid] = 0
                ClassFeatDict[classid][wid] += 1
            doc_num += 1
    print(ClassFeatDict)
    print(ClassFreq)

'''
在原有统计count做成概率形式
'''
def ComputeModel():
    global doc_num
    for classid in ClassFreq.keys():
        # 某类别的频率
        ClassProb[classid] = ClassFreq[classid]/float(doc_num)
    for classid in ClassFeatDict.keys():
        # 对未出现过的词进行平滑处理+1 (类别文章数+类别单词数)*0.1
        new_sum = float(ClassFreq[classid]+len(ClassFeatDict[classid])) * DefautFreq
        for wid in ClassFeatDict[classid].keys():
            ClassFeatDict[classid][wid] = (ClassFeatDict[classid][wid]+DefautFreq)/new_sum
        ClassDefaultProb[classid]=float(DefautFreq)/new_sum

### saveModel###
# p(y)
def SaveModel():
    with open(class_defaut,'w',encoding='utf-8') as f:
        f.write(str(ClassDefaultProb))
    with open(class_prob,'w',encoding='utf-8') as f:
        f.write(str(ClassProb))
    with open(class_feat,'w',encoding='utf-8') as f:
        f.write(str(ClassFeatDict))
    with open(word_dict,'w',encoding='utf-8') as f:
        f.write(str(WordDic))


def LoadModel():
    global ClassDefaultProb
    global ClassProb
    global ClassFeatDict
    global WordDic
    with open(class_defaut,'r',encoding='utf-8') as f:
        ClassDefaultProb = eval(f.read())
    with open(class_prob,'r',encoding='utf-8') as f:
        ClassProb = eval(f.read())
    with open(class_feat,'r',encoding='utf-8') as f:
        ClassFeatDict = eval(f.read())
    with open(word_dict, 'r', encoding='utf-8') as f:
        WordDic = eval(f.read())
    # return ClassDefaultProb,ClassProb,ClassFeatDict

# 预测各篇文章属于哪个类别
def predict():
    global ClassDefaultProb
    global ClassProb
    global ClassFeatDict
    global WordDic
    ture_label_list = []
    pred_label_list = []
    with open(TestOutFilePath,'r',encoding='utf-8') as f:
        for sline in f.readlines():  # 一行是一篇文章
            scoreDic = {}
            pos = sline.find('#')
            if pos > 0:
                sline = sline[:pos].strip()  # tag + words
            words = sline.split(' ')
            classid = int(words[0])
            ture_label_list.append(classid)
            words = words[1:]

            # 取p(y)
            for classid in ClassProb.keys(): # 每个类别都需要算一遍
                scoreDic[classid] = math.log(ClassProb[classid])

            # 取p（x|y）
            for word in words:
                if len(word)<1:
                    continue
                wid = int(word)
                # 这个词没有出现在整个训练词表中，直接丢掉
                if wid not in WordDic:
                    continue
                for classid in ClassFeatDict.keys():
                    # 这个词出现在此表中，但是没有出现在对应这个一个类别（classid）
                    # 表示这个词在别的classid出现过，计算概率的时候使用过
                    # 在没有出现这个词的类别中如果不使用这个词，
                    # 那就对于这两个类别比较概率的时候会出现不公平
                    if wid not in ClassFeatDict[classid]:
                        scoreDic[classid] += math.log(ClassDefaultProb[classid])
                    else:
                        scoreDic[classid] += math.log(ClassFeatDict[classid][wid])

            # 找出概率最大的类别
            maxProb = max(scoreDic.values())
            for classid in scoreDic.keys():
                if scoreDic[classid] == maxProb:
                    pred_label_list.append(classid)

        print(len(pred_label_list),len(ture_label_list))
        return ture_label_list,pred_label_list

def Evaluete(true_list,pred_list):
    accuracy = 0
    i = 0
    while i < len(true_list):
        if pred_list[i]==true_list[i]:
            accuracy+=1
        i+=1
    accuracy = float(accuracy)/float(len(true_list))
    print('Accuracy : ',accuracy)

if __name__ == '__main__':
    # LoadData()
    # ComputeModel()
    # SaveModel()
    LoadModel()
    true_list,pred_list = predict()
    Evaluete(true_list, pred_list)
    # print(ClassProb[0])





