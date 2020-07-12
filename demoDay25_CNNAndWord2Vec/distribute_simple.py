import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义标记，用在命令行执行时设置指定参数
flags = tf.app.flags
flags.DEFINE_string("data_dir","./tmp/mnist-data","mnist data path")
flags.DEFINE_integer("hidden_units",100,"Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps",100000,"Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size",100,"Training batch size")
flags.DEFINE_float("learning_rate",0.01,"Learning rate")
