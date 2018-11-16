import sys

import numpy as np
import tensorflow as tf

import nets.mobilenet_v1_1_224 as model
import heads.fc as head


tf.Graph().as_default()
sess = tf.Session()

batch_size = 1
width = 224
height = 224
channels = 3

images = tf.placeholder(
    tf.float32,
    shape=(batch_size, width, height, channels),
    name="input"
)

endpoints, body_prefix = model.endpoints(images, is_training=False)

with tf.name_scope('head'):
    endpoints = head.head(endpoints, is_training=False)

output = tf.identity(endpoints['age'], name='output')

saver = tf.train.Saver()
saver.restore(sess, './logs/checkpoint-99000')
saver.save(sess, "./logs/mobilenet.ckpt")
