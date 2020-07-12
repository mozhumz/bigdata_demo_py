# -*- coding: UTF-8 -*-
import os
import tensorflow as tf
from collections import Counter
import numpy as np
import time


# #####################单词数据进行编码转码的方法#########################
# 读取数据 ,返回word list
def read_word(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace('\n', '<eos>').split()


# 建立词典对，单词进行数字编码，对应数字可以理解单词在词表中的位置
def build_vocab(filename):
    data = read_word(filename)

    # word count
    counter = Counter(data)
    # 二次排序，先按照count，再按照单词  （word，count）
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


# 将文件进行编码
def file_to_word_ids(filename, word_to_id):
    data = read_word(filename)
    return [word_to_id[word] for word in data if word_to_id.get(word, -1) != -1]


# #####################定义输入数据结构#########################

def data_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, 'data_producer', [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.int32)

        data_len = tf.size(raw_data)  # 1280
        batch_len = data_len // batch_size   # 1280/128 = 10      128*10
        data = tf.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        # 如果不是>0,就是=0，等于0的需要报异常
        assertion = tf.assert_positive(
            epoch_size,
            message='epoch_size == 0,decrease batch_size or num_steps')

        # 控制依赖的上下文管理器（即需要assertion执行之后的epoch_size）
        with tf.control_dependencies([assertion]):
            # 返回一个新的tensor，这个tensor的内容和shape都和原来的一样，类似copy
            epoch_size = tf.identity(epoch_size, name='epoch_size')

        # 使用rang_input_producer多线程读取数据
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        # 提取tensor的一部分
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


class InputData(object):
    """The input data.模型处理数据输入数据"""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        # LSTM展开步数
        self.num_steps = num_steps = config.num_steps
        # 计算每个epoch中有多少轮的训练迭代，一个epoch相当于扫一遍数据集
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # 获取tensor格式的input_data和targets
        self.input_data, self.targets = data_producer(data, batch_size, num_steps, name=name)


# #####################定义LSTM模型#########################
class BaseModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size  # LSTM的节点数
        vocab_size = config.vocab_size  # 词汇表的大小

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        # 如果是训练状态且Dropout的keep_prob < 1，则前面的lstm_cell后面接一层Dropout层
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        # 使用RNN堆叠函数 将前面构造的lstm_cell多层堆叠得到cell ,堆叠次数 num_layers
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # 初始化状态为0
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        # 注意：LSTM单元可以读入一个单词并结合之前存储的状态state计算下一个单词出现的概率分布，
        # 并且每次读取一个单词后它的状态state会被更新
        with tf.device('/cpu:0'):
            # 行为词表大小，列为hidden_size,即为hidden_size个列向量【词】
            embedding = tf.get_variable('embedding', [vocab_size, size], dtype=tf.float32)
            # 查询单词对应向量表达获得inputs
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        # 如果为训练状态，添加一层dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.nn.rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):  # [batch_size,num_steps]
                # 因为第二次时state是已经有信息状态了，需要在后面重复更新使用，
                # 所以需要设置从第二次循环开始变量的复用
                if time_step > 0:
                    # 设置复用变量
                    tf.get_variable_scope().reuse_variables()
                # inputs[:, time_step, :]代表所有样本的第time_step个单词，得到输出和更新后的状态
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 最后输出层用全连接，一个向量形式
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])

        # 定义w,b
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

        # 套用公式加权求和
        logits = tf.matmul(output, softmax_w) + softmax_b

        # 损失，计算输出logits与targets的偏差，平均-logp损失
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        # 先用reduce_sum汇总整个batch的loss，然后平均到每个样本
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        # 如果不是训练（即为预测状态）状态直接从这返回
        if not is_training:
            return
        # 学习率变量learn rate：_lr
        self._lr = tf.Variable(0.0, trainable=False)
        #  获取全部可训练的参数
        tvars = tf.trainable_variables()
        # 基于前面得到的cost，计算tvars（变量）的梯度，
        # 设置梯度范式，即控制梯度在max_grad_norm范围内：防止梯度爆炸和梯度弥散
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        # 定义优化器
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # 将前面的梯度用到所有可训练的参数tvars上
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            # 生成全局统一的训练步数
            global_step=tf.contrib.framework.get_or_create_global_step())
        # 定义一个新的学习率，用来控制学习率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        # 将_new_lr赋值给_lr  <=> self._lr=self._new_lr
        self._lr_update = tf.assign(self._lr, self._new_lr)

    # 外部控制模型的学习速率，方式是将学习速率值传入到_new_lr这个placeholder中
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    # property装饰器可以将返回变量设置为只读，防止修改变量引发的问题
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


# #####################定义config参数#########################
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1  # 网络中权重值的初始scale
    learning_rate = 1.0  # 学习率的初始值
    max_grad_norm = 5  # 梯度的最大范数
    num_layers = 2  # LSTM可以堆叠的层数
    num_steps = 20  # LSTM梯度反向传播的展开步数
    hidden_size = 200  # 隐含节点数
    max_epoch = 4  # 初始学习速率可训练的epoch数
    max_max_epoch = 13  # 总共可训练的epoch数
    keep_prob = 1.0  # dropout层的保留节点的比例
    lr_decay = 0.5  # lr_decay是学习速率的衰减速度
    batch_size = 20  # 每个batch中样本的数量
    vocab_size = 10000  # 词表大小


# #####################定义训练过程#########################
def run_epoch(session, model, eval_op=None, verbose=False):
    """
    定义训练一个epoch数据的函数
    Runs the model on the given data.
    """
    start_time = time.time()
    # 初始化
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 输出结果字典，包含cost和final_state最终的状态
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }

    # 如果有评测操作也加入到fetches输出字典中
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    # 开始训练
    for step in range(model.input.epoch_size):
        feed_dict = {}
        # 将LSTM单元的state加入feed_dict中
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        # 进行一次训练就拿一次cost和final_state
        vals = session.run(fetches, feed_dict)   # sess.run({'cost':cost,'final':final_state},feed_dict={x:batch_x,y:batch_y})
        cost = vals["cost"]
        state = vals["final_state"]

        # 每次都进行累加
        costs += cost
        iters += model.input.num_steps

        # 完成10%的epoch就进行一次结果展示
        if verbose and step % (model.input.epoch_size // 10) == 10:
            # perplexity：平均cost的自然常数指数，比较nlp模型性能比较重要指标，越低test集上预测效果越好
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


if __name__ == '__main__':
    data_path = 'simple-examples/data/'
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    # 对数据中的单词编码 dict/MAP
    word_to_id = build_vocab(train_path)
    # 对文件进行转码
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    vocabulary_size = len(word_to_id)

    config = SmallConfig()
    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    # 创建默认的graph
    with tf.Graph().as_default():
        # 参数初始化在[-config.init_scale，config.init_scale]之间
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # 实例化模型
        with tf.name_scope("Train"):
            train_input = InputData(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = BaseModel(is_training=True, config=config, input_=train_input)
                # tf.scalar_summary("Training Loss", m.cost)
                # tf.scalar_summary("Learning Rate", m.lr)

        # 用来验证的模型
        with tf.name_scope("Valid"):
            valid_input = InputData(config, valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = BaseModel(is_training=False, config=config, input_=valid_input)
                # tf.scalar_summary("Validation Loss", mvalid.cost)
        # 测试的模型
        with tf.name_scope("Test"):
            test_input = InputData(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = BaseModel(is_training=False, config=eval_config,
                                  input_=test_input)
        # 创建训练的管理器sv
        sv = tf.train.Supervisor()
        # 创建默认session
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                # 累计的学习率衰减值，超出max_epoch的进行学习率衰减
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                # 更新学习率：初始学习率*累计的衰减
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
