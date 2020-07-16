# -*- coding: UTF-8 -*-
import tensorflow as tf2

print(tf2.__version__)
# 使用v1版本
tf = tf2.compat.v1
tf.disable_eager_execution()

import numpy as np
from sklearn.preprocessing import StandardScaler


# D(x)=(max-min)^2/12,均值：0，方差为(min+max)/2  [-sqrt(6/(max+min),sqrt(6/(max+min)))]
# 目标是要学习w（权重），快速求解w的值，做了简单w的初始化。
# 1. 层数多的时候，梯度会消失，w更新不大，
# 2. zero初始化，学习起来相对比较慢
def init_w(f_in, f_out, constant=1):
    low = -constant * np.sqrt(6 / (f_in + f_out))
    high = constant * np.sqrt(6 / (f_in + f_out))
    return tf.random_uniform((f_in, f_out), minval=low, maxval=high, dtype=tf.float32)


class AutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdadeltaOptimizer(),
                 scale=0.1):
        '''
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为softplus：log(exp(features) + 1)
        :param optimizer: 优化器，Adam
        :param scale: 高斯噪声系数，0.1
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model:定义网络结构#######################

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # sigmoid((x+噪音)*w1+b1)  softplus((x+噪音)*w1+b1) h*1
        self.hidden = self.transfer(tf.add(
            tf.matmul(self.x + scale * tf.random_uniform((n_input,)),self.weights['w1']),
            self.weights['b1']
            )
        )
        # 输出 = 隐含层【】*w2+b2 = x（reconstruction）w1:i*h x:i*1 w2:h*i h*1 i*1
        self.reconstruction = self.transfer(tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2']))

        # cost:定义损失函数： 1/2sum((输出-x)^2) 1*i i*1 1*1
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weight = dict()
        # 初始化权重在合适范围内：
        # 【初始化太小：在每层建传递更新较小，权重更新少，初始化权重太大，容易发散】
        all_weight['w1'] = tf.Variable(init_w(self.n_input, self.n_hidden))
        # b初始化
        all_weight['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        all_weight['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weight['b2'] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)
        return all_weight

    # 定义计算损失cost以及执行一步训练的函数，会触发训练
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 这个函数在模型训练结束后才调用，只计算最终的cost
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 返回隐含层的输出结果，也就是隐含层能学习出数据的高阶特征
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 以hidden层为输入，reconstruction为输出，自编码后半部分
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 获取w1
    def getWeight(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='G:\\bigdata\\badou\\00-data\\MNIST')


# mnist = input_data.read_data_sets('MNIST/', one_hot=True)

# 对训练数据和test数据做标准化，调用sklearn中对应的标准化方法，0均值，1方差
def standard_scale(X_train, X_test):
    preprocessor = StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# (X_train,y_train),(X_test,y_test )= standard_scale(mnist.train.images,mnist.test.images)

n_samples = X_train.shape[0]
# 扫多少次数据集，迭代几次
training_epochs = 50
batch_size = 128

auto_encoder = AutoEncoder(n_input=784,  # 28*28
                           n_hidden=2000,
                           transfer_function=tf.nn.sigmoid,
                           optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.2),
                           scale=0.01)


# 有放回的采样，以采样一个batch_size大小的数据
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)  # batch_size = 2, len(data)=10  [0,9]
    return data[start_index:(start_index + batch_size)]


feat = []
# 开始训练
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)  # 1000 10   100个10条
    # loop 循环所有的batch
    for i in range(total_batch):
        # sample batch
        batch_xs = get_random_block_from_data(X_train, batch_size)
        new_xs = np.reshape(batch_xs, newshape=(batch_xs.shape[0], batch_xs.shape[1] * batch_xs.shape[2]))
        # fit training using batch data
        cost = auto_encoder.partial_fit(new_xs)
        avg_cost += cost * batch_size / n_samples

        if epoch == training_epochs - 1:
            hidden_feat = auto_encoder.transform(new_xs)
            feat.append(hidden_feat)

    if epoch % 1 == 0:
        print()
        print('Epoch: %04d' % (epoch + 1), 'training_cost={:.9f}'.format(avg_cost))

print('Total test cost: ' + str(
    auto_encoder.calc_total_cost(np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1] * X_test.shape[2])))))
print(np.array(feat[0]).shape)
