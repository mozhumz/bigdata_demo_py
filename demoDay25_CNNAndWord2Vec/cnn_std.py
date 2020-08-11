# -*- coding: UTF-8 -*-
# import tensorflow as tf2
# 使用v1版本
# tf = tf2.compat.v1
# tf.disable_eager_execution()
# from __future__ import absolute_import, division, print_function, unicode_literals
# tf2版本
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus, cpus)
tf.config.experimental.set_visible_devices(devices=gpus[2:4], device_type='GPU')

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data("F:\\八斗学院\\视频\\14期正式课\\00-data\\mnist2/")
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
# 特征缩放[0, 1]区间
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.summary() # 显示模型的架构

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary() # 显示模型的架构

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5)
# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:'+str(test_acc)+',test_loss:'+str(test_loss))
# 0.9873999953269958