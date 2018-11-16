import tensorflow as tf

from nets.mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
from tensorflow.contrib import slim


def endpoints(image, is_training):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, 3]')

    with tf.contrib.slim.arg_scope(mobilenet_v1_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = mobilenet_v1(image, num_classes=1001, is_training=is_training)

    endpoints['reduce_dims'] = tf.squeeze(endpoints['AvgPool_1a'], [1,2], name='reduce_dims')

    return endpoints, 'MobilenetV1'




