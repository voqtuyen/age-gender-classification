from argparse import ArgumentParser
import sys
import os
import time
import logging.config

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from nets import mobilenet_v1_1_224 as model
from heads import fc as head
from utils import common

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = ArgumentParser(description='Train a multi-task age/gender classification')

parser.add_argument('--train_dir', default='./logs', help='dir to store trainning checkpoints and logs')
parser.add_argument('--init_ckpt', default='./checkpoint/mobilenet_v1_1.0_224.ckpt', help='path to pretrained models')
parser.add_argument('--batch_size', default=32, help='training batch size')
parser.add_argument('--image_root', default='./dataset', help='image directory')
parser.add_argument('--csv_file', default='./data/age_gender_train.csv', help='image directory')
parser.add_argument('--learning_rate', default=0.001, help='learning rate')
parser.add_argument('--train_iters', default=100000, help='number of training iterations')
parser.add_argument('--decay_start_iter', default=20000, help='start decay iteration')
parser.add_argument('--ckpt_freq', default=1000, help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint')
parser.add_argument('--input_width', default=224, help='Network input width')
parser.add_argument('--input_height', default=224, help='Network input height')
parser.add_argument('--loading_threads', default=8, help='Number of loading threads')

def main():
    args = parser.parse_args()

    if os._exists(args.train_dir):
        print('The directory {} already exists'.format(args.train_dir))
        exit(1)
    else:
        os.makedirs(args.train_dir)
    
    # log_file = os.path.join(args.train_dir, "log")
    # logging.config.dictConfig(common.get_logging_dict(log_file))
    # log = logging.getLogger('train')

    # # Also show all parameter values at the start, for ease of reading logs.
    # log.info('Training using the following parameters:')
    # for key, value in sorted(vars(args).items()):
    #     log.info('{}: {}'.format(key, value))

    # Load the data from the CSV file.
    gender_lbls, age_labels, fids = common.load_dataset(args.csv_file, args.image_root)
    dataset = tf.data.Dataset.from_tensor_slices((fids, gender_lbls, age_labels))
    dataset = dataset.shuffle(len(gender_lbls))

    # Constrain the dataset size to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.
    dataset = dataset.take((len(gender_lbls) // args.batch_size) * args.batch_size)
    dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.

    dataset = dataset.map(
        lambda fid, gender_lbl, age_lbl: common.fid_to_image(
            fid, gender_lbl, age_lbl, image_root=args.image_root,
            image_size=(args.input_width, args.input_height)),
        num_parallel_calls=args.loading_threads)
    
    dataset = dataset.batch(args.batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    images, fids, gender_lbls, age_lbls = dataset.make_one_shot_iterator().get_next()

    endpoints, body_prefix = model.endpoints(images, is_training=True)

    with tf.name_scope('head'):
        endpoints = head.head(endpoints, is_training=True, dropout_keep_prob=0.5)

    ages = endpoints['age_gender'][:,0:9]
    genders = endpoints['age_gender'][:,9:]    

    gender_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=genders, labels=gender_lbls)
    age_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ages, labels=age_lbls)

    loss_mean = tf.reduce_mean(gender_losses) + tf.reduce_mean(age_losses)

    model_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    if 0 <= args.decay_start_iter and args.decay_start_iter < args.train_iters:
        learning_rate = tf.train.exponential_decay(args.learning_rate, tf.maximum(0, global_step - args.decay_start_iter),
            args.train_iters - args.decay_start_iter, 0.0001)
    else:
        learning_rate = learning_rate

    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss_mean, global_step=global_step)

    checkpoint_saver = tf.train.Saver(max_to_keep=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model_variables)
        saver.restore(sess, args.init_ckpt)

        for i in range(args.train_iters):
            start_time = time.time()
            _, b_loss, step = sess.run([train_op, loss_mean, global_step])
            elapsed_time = time.time() - start_time
            seconds_to_do = round((args.train_iters - step) * elapsed_time / 3600,2)
            print('Iter: ' + str(step) + ' | loss: ' + str(b_loss) + ' | ETA: ' + str(seconds_to_do))

            if (args.ckpt_freq > 0 and step % args.ckpt_freq == 0):
                checkpoint_saver.save(sess, os.path.join(args.train_dir, 'checkpoint'), global_step=step)

        checkpoint_saver.save(sess, os.path.join(args.train_dir, 'checkpoint'), global_step=step)

if __name__ == "__main__":
    main()
