# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST/', one_hot=True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf.Session()


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 1步长（stride size），0边距（padding size）的模板，保证输出和输入时同一个大小
# padding:是否对输入的图像矩阵边缘补0，'SAME'（是）和'VALID'（否）
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


# 卷积核大小5*5 -> 32 feature map , 1表示channel，黑白
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 取到的是one_hot形式的图片，即在一个数组中
x = tf.placeholder(tf.float32, shape=[None, 784])
# reshape将数组转化成28*28矩阵大小，即图片
'''
这里是将一组图像矩阵x重建为新的矩阵，该新矩阵的维数为（a，28，28，1），其中-1表示a由实际情况来定。 
例如，x是一组图像的矩阵（假设是50张，大小为56×56），则执行x_image = tf.reshape(x, [-1, 28, 28, 1])
可以计算a=50×56×56/28/28/1=200。即x_image的维数为（200，28，28，1）
'''
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ############### model #################

# 第一层：以relu为激励函数（与softplus，sigmoid一样的功能），前向传播进行（加权+偏置bias）， 卷积，池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 32个28*28 feature maps
# 12*12
h_pool1 = max_pool_2x2(h_conv1)  # 32个14*14 feature maps

# 第二层：这里为什么是channel=32？
# 因为上一层生成了32个feature map，channel=1（原始图片channel）*32（feature map数量） 64表示要生成的feature map数量
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 64@14*14
h_pool2 = max_pool_2x2(h_conv2)  # 64@7*7

# 第三层：7*7 图片大小，因为需要做全连接层，需要将图片（矩阵）变成向量 shape
# 还有没有别的方式进行这一层得到全连接层？
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# 1. 按照一定概率丢失节点，可以减少过拟合
# 2. 自动处理输出值的scale
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练设置
y_ = tf.placeholder(tf.float32, shape=[None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 预测设置 index[0,1,2,3,4,5,6,7,8,9] [0.1,0.01,0.03,0.05,0.5,0.2,0.01,...]  [0,0,0,0,1,0,0,0,0,...]
correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # [0,0,1,1,1]
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess.run(tf.initialize_all_variables())

# 开始训练打印日志
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 1000 == 0:
        # 预测过程需要所有节点都起到作用，所以keep_prob:1.0
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step: %d,training accuracy %g' % (i, train_accuracy))
    # 训练过程需要将部分节点dropout，所以要给定一个概率0.5  early_stop
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('Test accuracy %g' % accuracy.eval(session=sess, feed_dict={x: mnist.test.images,
                                                                      y_: mnist.test.labels,
                                                                      keep_prob: 1.0}))
