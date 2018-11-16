import tensorflow as tf
from tensorflow.contrib import slim



def head(endpoints, is_training, dropout_keep_prob=0.5, num_age_classes=9, num_gender_classes=2):

    drop_out = slim.dropout(endpoints['reduce_dims'], keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_1b')

    endpoints['head_output'] = slim.fully_connected(
        drop_out, 1024, normalizer_fn=slim.batch_norm,

        activation_fn=tf.nn.relu,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        })

    endpoints['age_gender'] = slim.fully_connected(
        endpoints['head_output'], num_age_classes + num_gender_classes, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='Age_gender')

    return endpoints
