import tensorflow as tf
import numpy as np
import os
from architecture import ResNetAE4_class as ResNet
import time
from datetime import datetime
from ops import get_data_moving_MNIST, make_tensorboard, clf_optimization


def run_inference(args):
    # images = get_data_moving_MNIST(args)
    images = np.load(os.path.join(args.data_path,'moving_mnist_test_with_labels.npz'))
    images = images['arr_0']
    images = np.transpose(images,axes=[0,3,2,1])
    _images = tf.placeholder(tf.float32)
    x_in = tf.reshape(_images, [1, 64, 64, 1])
    x_in = (tf.cast(x_in, tf.float32)) / 255.

    data_constraint = 0.9

    x_in = 2. * tf.cast(x_in, tf.float32)  # [0, 2]
    x_in -= 1.  # [-1, 1]
    x_in *= data_constraint  # [-.9, .9]
    x_in += 1.  # [.1, 1.9]
    x_in /= 2.  # [.05, .95]
    with tf.variable_scope('autoencoder'):
        ae_model = ResNet(z_dim=args.latent_dim, n_levels=args.n_levels)
        with tf.variable_scope('encoder'):
            z_in = ae_model.encoder(x_in)
        with tf.variable_scope('decoder'):
            im_out = ae_model.decoder(z_in)

    step = tf.get_variable(
        'global_step', [], tf.int64,
        tf.zeros_initializer(),
        trainable=False)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)))

    initialized = False
    previous_global_step = 0  # don"t run eval for step = 0

    ckpt_state = tf.train.get_checkpoint_state(args.logdir)
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    i = 0
    images_out = np.zeros((images.shape[0], 8, 8, 1))
    for img in images:
        print(i)
        img = img[tf.newaxis,...]
        outputs = sess.run(z_in , feed_dict={_images : img})
        images_out[i] = outputs
        i += 1
    np.save('predicted_latent_space_movingMNIST_with_Labels.npy',images_out)


def run_train(args):
    images, labels = get_data_moving_MNIST(args)
    labels = tf.reshape(labels, [args.batch_size,])
    labels = tf.one_hot(labels, 10)
    x_in = tf.reshape(images, [args.batch_size, 64, 64, 1])
    x_in = (tf.cast(x_in, tf.float32)) / 256.

    data_constraint = 0.9

    x_in = 2. * tf.cast(x_in, tf.float32)  # [0, 2]
    x_in -= 1.  # [-1, 1]
    x_in *= data_constraint  # [-.9, .9]
    x_in += 1.  # [.1, 1.9]
    x_in /= 2.  # [.05, .95]

    with tf.variable_scope('classifier'):
        ae_model = ResNet(num_of_classes=10, z_dim=args.latent_dim, n_levels=args.n_levels)
        with tf.variable_scope('encoder'):
            class_out = ae_model.encoder(x_in)

    config = tf.ConfigProto(
        # log_device_placement=True,
        # allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    loss, step, train_step, global_step = clf_optimization(args, labels, class_out)
    summary_op = make_tensorboard(x_in,class_out, loss, args)

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:

        if args.resume:
            saver.restore(sess, tf.train.latest_checkpoint(args.logdir))
            step_init = sess.run(global_step)
        else:
            initializer = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
            sess.run(initializer)
            step_init = 0

        train_writer = tf.summary.FileWriter(args.logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(step_init,int(args.train_steps)):
                start_time = time.time()

                summary , _loss, _ = sess.run([summary_op, loss,train_step])
                duration = time.time() - start_time

                if step % 100 == 0:

                    train_writer.add_summary(summary, step)
                    examples_per_sec = args.batch_size / float(duration)
                    format_str = ('%s: step %d, ae_loss = %.5f, '
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
    parser.add_argument('--name', default='resae_classifier_z1_nlevels3_movingmnist')
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--window_size', default=1)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--latent_dim', default=1)
    parser.add_argument('--n_levels', default=3)

    parser.add_argument('--train_steps', default=20000000)

    parser.add_argument('--data_path', default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data')
    parser.add_argument('--base_logdir',
                        default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/logs')
    parser.add_argument('--resume', default=False)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
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
    args.learning_rate = 0.0001
    args.clip_gradient = 100.
    args.image_size = 64
    args.classifier = True

    if args.train:
        run_train(args=args)
    else:
        run_inference(args=args)