# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
tf.reset_default_graph()


# # 读入数据
mnist = input_data.read_data_sets('MNIST/', one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)  # (55000, 784) (55000, 10)
print(mnist.test.images.shape,mnist.test.labels.shape)    # (10000, 784) (10000, 10)
print(mnist.validation.images.shape,mnist.validation.labels.shape)  # (5000, 784) (5000, 10)

def  img_show(img):
    img = Image.fromarray(im * 255)  # Image和Ndarray互相转换
    img = img.convert('RGB')  # jpg可以是RGB模式，也可以是CMYK模式
    plt.imshow(im, cmap="gray")
    plt.show()

im = mnist.train.images[2].reshape((28, 28))  # 读取的格式为Ndarry
print(mnist.train.labels[2])
img_show(im)

#
# 定义原本空间
# x = tf.placeholder(tf.float32,[None,784])
print(type(mnist.train.images))
#
# # 定义参数,并初始化  类别【0-9】有10个
# W = tfe.Variable(tf.zeros([784,10])) # 784 = 28*28
# b = tfe.Variable(tf.zeros([10]))
#
# # 定义model：x经过加权求和后做softmax非线性变化，得到类别概率 [正向传播] p(y=label1|X)
# y = tf.nn.softmax(tf.matmul(mnist.train.images,W)+b)  # pred
#
# a = [[0,0,0,1,0,0],[1,0,0,0,0,0]]
# print(tf.argmax(a,axis=1))  # tf.Tensor([3 0], shape=(2,), dtype=int64)
# print(tf.argmax(a,axis=0))  # tf.Tensor([1 0 0 0 0 0], shape=(6,), dtype=int64)

# a = tf.truncated_normal([3,3], stddev=0.1)
# y_ = [[0,0,1],[0,1,0]]
# y_conv = [[0.2,0.1,0.7],[0,0.8,0.2]]
#
#
# with tf.Session() as sess:
#     print(sess.run(tf.argmax(y_,1)))  # [2 1]
#     print(sess.run(tf.argmax(y_conv, 1))) # [2 1]
#     print(sess.run(tf.equal(tf.argmax(y_,1),tf.argmax(y_conv, 1))))  # [ True  True]
#     print(sess.run(tf.cast(tf.equal(tf.argmax(y_,1),tf.argmax(y_conv, 1)),tf.float32)))  # [1. 1.]
# valid_size = 16
# # 验证单词从频率最高的100个单词中抽取
# valid_window = 100
# # 随机抽取验证样本集
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# print(valid_examples)


# embedding_weight = tf.Variable([1,2,3,4,5,6])
# x = tf.Variable([0,0,0,1,1,0])
# x_feat_ind = tf.Variable([3,4])
# x_feat_val = tf.Variable([1,1])
#
# weight_out = tf.nn.embedding_lookup(embedding_weight,x_feat_ind)
# reshape_xval = tf.reshape(x_feat_val,shape=[-1,2,1]) # 把行向量x转变成列向量
# w_X_x = tf.multiply(weight_out,x_feat_val)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(weight_out))  # [4 5]
#     print(sess.run(reshape_xval))  # [[[1],[1]]]
#     print(sess.run(w_X_x))  # [4 5]