# -*- coding: UTF-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST/', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)  # (55000, 784) (55000, 10)
print(mnist.test.images.shape, mnist.test.labels.shape)    # (10000, 784) (10000, 10)
print(mnist.validation.images.shape, mnist.validation.labels.shape)  # (5000, 784) (5000, 10)

# 定义session
sess = tf.InteractiveSession()
# 定义原本空间
x = tf.placeholder(tf.float32, [None, 784])

# 定义参数,并初始化  类别【0-9】有10个
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 定义model：x经过加权求和后做softmax非线性变化，得到类别概率 [正向传播] p(y=label1|X)
y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))  # pred

# 给真实标签占位置
y_ = tf.placeholder(tf.float32, [None, 10])  # true

# 定义交叉熵为损失函数loss，定义怎么来算误差的
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

# 定义优化方式，BP，怎么求参数 【用梯度下降法反向传播求参数，目标最小化损失函数cross_entropy】
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(cross_entropy)

# [0,1,0,1,1,1,1,1,1,1] tf.argmax(y,1) index类别 [0.1,0.2,0.01,0.08,0.01,0.2,0.4,0,0,0] => 6
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))  # [0,0,0,0,0,0,1,0,0,0] =>6   1
# 所有为1求和/样本个数  8/10 => sum(correct_pred)/len(correct_pred) = avg => reduce_mean
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化参数
tf.global_variables_initializer().run()
batch_size = 100  # dataset
n_batch = mnist.train.num_examples // batch_size
# 训练的过程,得到参数w，b
for i in range(30):  # epoch
    # 扫一遍数据集
    for batch in range(n_batch):
        # 每次取100条数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})

    train_acc = sess.run(accuracy,feed_dict={x: mnist.train.images,y_: mnist.train.labels})
    test_acc = sess.run(accuracy,feed_dict={x: mnist.test.images,y_: mnist.test.labels})
    print('step%s: train accuracy: %s,test accuracy: %s'%(i,train_acc,test_acc))






