# -*- coding: UTF-8 -*-

import collections
import math
import os
import random
import zipfile

import numpy as np
import urllib
import tensorflow as tf

# Step 1 :下载数据
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    return filename


filename = maybe_download('text8.zip')


# 读取数据到list
def read_Data(filename):
    with zipfile.ZipFile(filename) as f:
        # 将数据转化成单词列表
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_Data(filename)
print(words)
print('Data Size', len(words))

# Step 2:建立字典，对word进行编码，并提取unkown单词
vocabulary_size = 50000


def build_dataset(words):  # 一对单词
    count = [['UNK', 1]]
    # Counter统计word count，用most_common取前50000个单词为词表
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    # words = set()
    # d = dict(zip(words,range(len(words))))
    # 常用的word编码方式
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0  # 不会出现在词表中有多少个词
    # 将所有单词填回到原数据中
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # 建立一个字典：通过index查找单词
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
# 为了节省空间
del words
print('Most common words(+UNK)', count[:5])
# 看下样本数据和reverse_dictionary字典的数据
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# Step 3: 生成skip-gram model的训练batch #################
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    '''
    :param batch_size: batch大小
    :param num_skips: 对每个单词生成多少个样本
    :param skip_window:单词最远可联系的距离
    :return: batch，对应labels
    '''
    # 需要在这个函数中反复修改，定义为global
    global data_index
    assert batch_size % num_skips == 0  # 4 batch_size=17 4个目标单词生成16条数据
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [skip_window target skip_window]
    # 创建一个最大容量为span的deque，即双向队列
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        # 只会保留最后插入的span个变量
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # 第一层循环，每次循环内对一个目标单词生成样本，buffer中是目标单词和所有相关的词
    for i in range(batch_size // num_skips):  # 多少个目标单词
        # 目标单词
        target = skip_window
        targets_to_avoid = [skip_window]
        # 除了目标之外的词
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            # 其中一条样本的feature是target
            batch[i * num_skips + j] = buffer[skip_window]
            # 标签是语境word,当前的target和buffer中skip_window取到的不同
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=9, num_skips=3, skip_window=2)
for i in range(9):
    # wid,word -> label wid,label word
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: 定义skip-gram model 并进行训练

batch_size = 128
# 单词转化为稠密向量的维度
embedding_size = 128
skip_window = 1
num_skips = 2

# 抽取验证单词数量
valid_size = 16
# 验证单词从频率最高的100个单词中抽取
valid_window = 100
# 随机抽取验证样本集
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 训练时用来做负样本的噪声单词的数量
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    # Input Data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # with tf.device('/cpu:0'):
    with tf.device('/gpu:0'):
        # 初始化词向量：随机生成所有单词的词向量embeddings,（weight初始化）
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 查找输入train_inputs对应的向量embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # 使用NCE（噪声对比估计训练损失） loss作为优化目标，初始化weights使用（截断正态分布）和biases
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 定义损失
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))
    # 定义优化方法
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    # 占比概率形式，归一化 norm算embedding的模大小
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    # 查询验证单词的嵌入向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # 验证单词的嵌入向量与词汇表中的所有单词的相似度
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # 变量初始化
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001
# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。GPU需要加上tf.ConfigProto()
# with tf.Session(graph=graph) as session:  # cpu使用
with tf.Session(graph=graph,
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:  # GPU使用
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # 每10000次，计算一次验证单词与全部单词的相似度，并将与每个验证单词最相似的8个单词展示出来。
        if step % 10000 == 0:
            sim = similarity.eval()
            print(sim)
            for i in range(valid_size):
                # 验证词的index从字典取出对用的词
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                # 取到相似的top8个单词的index
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                # 通过相似单词的index从字典中取到对应的词
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    # 最终embedding的矩阵，也就是所有单词对应的（weight矩阵）vector
    final_embeddings = normalized_embeddings.eval()
