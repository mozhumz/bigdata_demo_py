import tensorflow as tf

initial = tf.truncated_normal((2,2), stddev=0.1)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    print(sess.run(initial))

