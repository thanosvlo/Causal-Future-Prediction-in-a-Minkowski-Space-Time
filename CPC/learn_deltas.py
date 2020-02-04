import tensorflow as tf
import os
import numpy as np
from architecture import ResNetAE4 as ResNet
from architecture import DeltaModel as DeltaModel

import time
from datetime import datetime
from delta_ops import  get_data_moving_MNIST_sequences,optimization_func

# tf.executing_eagerly()

def run_train(args):
    images, test_image, labels, test_labels, times = get_data_moving_MNIST_sequences(args)
    x_in = tf.reshape(images, [-1, 64, 64, 1])
    x_tests = tf.reshape(test_image, [-1, 64, 64, 1])
    # times = tf.reshape(times[:, 0], [-1, 1])
    # times = tf.cast(times, 'float32')
    times = tf.tile(times, [1,64])
    times = tf.reshape(times, [-1,8, 8, 1])

    data_constraint = 0.9

    x_in = 2. * tf.cast(x_in, tf.float32)  # [0, 2]
    x_in -= 1.  # [-1, 1]
    x_in *= data_constraint  # [-.9, .9]
    x_in += 1.  # [.1, 1.9]
    x_in /= 2.  # [.05, .95]

    x_tests = 2. * tf.cast(x_tests, tf.float32)  # [0, 2]
    x_tests -= 1.  # [-1, 1]
    x_tests *= data_constraint  # [-.9, .9]
    x_tests += 1.  # [.1, 1.9]
    x_tests /= 2.  # [.05, .95]

    with tf.variable_scope('autoencoder'):
        ae_model = ResNet(z_dim=args.latent_dim, n_levels=args.n_levels)
        with tf.variable_scope('encoder'):
            z_in = ae_model.encoder(x_in)
        with tf.variable_scope('decoder'):
            im_out = ae_model.decoder(z_in)


    # Now lets go for the deltas
    z_tests = ae_model.encoder(x_tests)
    z_tests = tf.layers.batch_normalization(z_tests)
    z_in = tf.layers.batch_normalization(z_in)

    with tf.variable_scope('delta_prediction'):
        delta_model = DeltaModel(args)
        deltas = delta_model.encode(z_in)


    # Loss function

    # loss, train_step = optimization_func(args, z_in, z_tests, deltas, test_labels, times)



    # Triangularize L
    # operator = tf.linalg.LinearOperatorLowerTriangular(deltas)
    # deltas = operator.to_dense()

    #  L * (t-t0)

    L = tf.math.multiply(times, deltas)


    # x-c
    z_tests = z_tests - z_in


    incled_1 = tf.math.divide(z_tests, L+1e-6)
    # incled_2 = tf.reduce_sum(incled_1, axis=(1,2,3))
    incled = tf.norm(incled_1,axis=(1,2))


    # the negatives x which are included are casted to 0 and then incls will be 1 the correct label
    # incls = incled[..., tf.newaxis]
    incls = incled - 1
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
    grads_and_vars = optimizer.compute_gradients(
        loss, tf.trainable_variables(scope='delta_prediction'))
    grads, vars_ = zip(*grads_and_vars)
    capped_grads, gradient_norm = tf.clip_by_global_norm(
        grads, clip_norm=args.clip_gradient)

    capped_grads_and_vars = zip(capped_grads, vars_)
    train_step = optimizer.apply_gradients(
        capped_grads_and_vars, global_step=step)

    # Restore AE weights
    saver_ae = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
    sess = tf.Session(config=tf.ConfigProto(
        # allow_soft_placement=True,
        # log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)))

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    ckpt_state = tf.train.get_checkpoint_state(args.logdir_transfer)
    if ckpt_state and ckpt_state.model_checkpoint_path:
        print("Loading file %s" % ckpt_state.model_checkpoint_path)
        saver_ae.restore(sess, ckpt_state.model_checkpoint_path)

    # Saver for new model
    saver = tf.train.Saver()

    # train_writer = tf.summary.FileWriter(args.logdir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step_init = 0


    # Begin Training
    try:
        print('TRAINING...')
        for step in range(step_init,int(args.train_steps)):
            start_time = time.time()

            _loss,_ = sess.run([loss, train_step])
            duration = time.time() - start_time

            if step % 100 == 0:

                # train_writer.add_summary(summary, step)
                examples_per_sec = args.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.5f, '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, _loss,
                                    examples_per_sec, duration))

            if step % 1000 == 0 or (step + 1) == args.train_steps:
                       checkpoint_path = os.path.join(args.logdir, 'iter_{}.ckpt'.format(step))
                       saver.save(
                           sess,
                           checkpoint_path,
                           global_step=step)
    except tf.errors.OutOfRangeError:
        # End of dataset
        tf.logging.log(tf.logging.INFO, 'End of Training')

    except KeyboardInterrupt:
        tf.logging.log(tf.logging.INFO, 'Keyboard Interrupt!')

    finally:
        tf.logging.log(tf.logging.INFO, 'Stopping Threads')
        coord.request_stop()
        coord.join(threads)
        tf.logging.log(tf.logging.INFO, 'Saving iter: {}'.format(step))
        saver.save(sess, os.path.join(args.logdir, 'iter_{}.ckpt'.format(step)))




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dev')
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--window_size', default=1)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--latent_dim', default=1)
    parser.add_argument('--n_levels', default=3)
    parser.add_argument('--positive', default=5)
    parser.add_argument('--negative', default=5)


    parser.add_argument('--train_steps', default=20000000)

    parser.add_argument('--data_path', default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data')
    parser.add_argument('--logdir_transfer', default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/logs/resae_z1_nlevels3_movingmnist')
    parser.add_argument('--base_logdir',
                        default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/logs')
    parser.add_argument('--resume', default=False)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--train', default=True)
    args = parser.parse_args()

    print(args.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.logdir = os.path.join(args.base_logdir, args.name) + '/'

    if args.train:
        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)
    tf.set_random_seed(0)

    # optimizer hyperparameters
    args.momentum = 1e-1
    args.decay = 1e-3
    args.learning_rate = 0.001
    args.clip_gradient = 100.
    args.image_size = 64

    if args.train:
        run_train(args=args)
    else:
        pass
        # run_inference(args=args)