#coding:utf-8
import tensorflow as tf
a=tf.constant([1.0,2.0,3.0],shape=[3],name='a')
b=tf.constant([1.0,2.0,3.0],shape=[3],name='b')
c=a+b
#log_device_placement=True 输出运行每一个运算的设备
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))

# initial = tf.truncated_normal((2,2), stddev=0.1)

# with tf.Session() as sess:
#     # sess.run(tf.global_variables_initializer())
#     print(sess.run(initial))

    # import tensorflow as tf

# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
#     a = tf.constant(1)
#     b = tf.constant(3)
#     c = a + b
#     print('结果是：%d\n 值为：%d' % (sess.run(c), sess.run(c)))


