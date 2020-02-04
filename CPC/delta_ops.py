import numpy as np
import tensorflow as tf
import os
import glob
import sympy
from sympy import Poly


def get_data_moving_MNIST_sequences(args):
    device = "/cpu:0"
    with tf.device(
            tf.train.replica_device_setter(0, worker_device=device)):
        def _parse_function(serialized_example):
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'data': tf.FixedLenFeature([], tf.string),
                    'test_data': tf.FixedLenFeature([], tf.string),
                    'test_labels': tf.FixedLenFeature([], tf.int64),
                    'times': tf.FixedLenFeature([], tf.int64),
                    'labels': tf.FixedLenFeature([], tf.int64),
                })
            image = features['data']
            test_image = features['test_data']
            labels = features['labels']
            test_labels = features['test_labels']
            times = features['times']

            image = tf.decode_raw(image, tf.uint8)
            test_image = tf.decode_raw(test_image, tf.uint8)

            labels = tf.cast(labels, tf.float32)
            test_labels = tf.cast(test_labels, tf.float32)
            times = tf.cast(times, tf.float32)

            test_image = tf.reshape(test_image, [64, 64, 1])
            image = tf.reshape(image, [64, 64, 1])
            labels = tf.reshape(labels, [1,])
            test_labels = tf.reshape(test_labels, [1,])
            times = tf.reshape(times, [1,])
            return image, test_image, labels, test_labels, times

        # dataset = tf.data.TFRecordDataset(gfile.Glob(FLAGS.data_path+'/*'))
        if args.train:
            print('Train files:', glob.glob(os.path.join(args.data_path , 'train_with_labels_sequence_pos_neg.tfrecord')))
            files = tf.data.Dataset.list_files(glob.glob(os.path.join(args.data_path , 'train_with_labels_sequence_pos_neg.tfrecord')))
        else:
            print('Test files:', glob.glob(os.path.join(args.data_path, 'test_with_labels_sequence_pos_neg.tfrecord')))
            files = tf.data.Dataset.list_files(glob.glob(os.path.join(args.data_path, 'test_with_labels_sequence_pos_neg.tfrecord')))

        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=100, block_length=20)
        dataset = dataset.map(_parse_function)  # Parse the record into tensors.
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.shuffle(5000)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(1)
        images, test_image, labels, test_labels, times = dataset.make_one_shot_iterator().get_next()
        return images, test_image, labels, test_labels, times

#
# def inclusion_test(args,x,ts,ds):
#     # positive_frames, negative_frames = tf.split(x, [args.postive,args.negative], axis=0)
#     # positive_frames = tf.split(positive_frames, args.positive, axis=0)
#     # negative_frames = tf.split(negative_frames, args.negative, axis=0)
#
#     x_minus_centers = tf.subtract(x,)
#
#
#     # xs = sympy.symbols('x0:%d' % x.shape[1])
#
#     time_difference = ts -
#     L_t = ds * time_difference
#     x_c =


def optimization_func(args, x, x_tests, deltas, test_labels, times):

    # Triangularize L
    operator = tf.linalg.LinearOperatorLowerTriangular(deltas)
    deltas = operator.to_dense()

    #  L * (t-t0)

    L = tf.math.multiply(times, deltas)

    # x-c
    z_tests = z_tests - z_in

    # || L * (t-t0)  dot  x-c || is included if <=1
    incled_1 = tf.matmul(L, z_tests, transpose_a=True)
    # incled_2 = tf.reduce_sum(incled_1, axis=(1,2,3))
    incled = tf.norm(incled_1[:, :, 0, 0], axis=1)

    # the negatives x which are included are casted to 0 and then incls will be 1 the correct label
    incls = incled[..., tf.newaxis]
    incls = incls - 1
    incls = tf.math.sign(incls)
    incls = tf.abs(incls - 1) / 2
    # # incls = tf.clip_by_value(incls, 0., 1.)
    # # incls = tf.abs(1 - incls)
    # # incls = tf.sigmoid(incls)  # set all to 0 or 1
    # incls = tf.nn.softmax(1 - incls)

    loss = tf.losses.sigmoid_cross_entropy(test_labels, incls)

    # optimization below

    momentum = 1. - args.momentum
    decay = 1. - args.decay
    optimizer = tf.train.AdamOptimizer(
        learning_rate=args.learning_rate,
        beta1=momentum, beta2=decay, epsilon=1e-08,
        use_locking=False, name="Adam")

    step = tf.get_variable(
        "delta_step", [], tf.int64,
        tf.zeros_initializer(),
        trainable=False)
    train_step = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                    scope='delta_prediction'))
    # grads_and_vars = optimizer.compute_gradients(
    #     loss, tf.trainable_variables(scope='delta_prediction'))
    # grads, vars_ = zip(*grads_and_vars)
    # capped_grads, gradient_norm = tf.clip_by_global_norm(
    #     grads, clip_norm=args.clip_gradient)
    #
    # capped_grads_and_vars = zip(capped_grads, vars_)
    # train_step = optimizer.apply_gradients(
    #     capped_grads_and_vars, global_step=step)

    return loss, train_step
