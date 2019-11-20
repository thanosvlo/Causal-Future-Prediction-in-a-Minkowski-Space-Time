import numpy as np
import os
import tensorflow as tf
from architecture import ResNetAE4 as ResNet
from sympy import Poly, Intersection
from sympy.abc import x
import sympy
from sympy.solvers.inequalities import solve_poly_inequality
from sympy.solvers import solve

import mystic.symbolic as ms


def load_model(args):
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


    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)))


    ckpt_state = tf.train.get_checkpoint_state(args.logdir)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    return im_out


def run_hyperspheres(args):

    # load test dataset
    image_data = np.load(os.path.join(args.data_path,'moving_mnist_test_with_labels.npz'))
    image_data = image_data['arr_0']
    image_data = np.transpose(image_data,axes=[0,3,2,1])
    image_data = image_data / 255.



    data = np.load(os.path.join(args.data_path,'predicted_latent_space_movingMNIST_with_Labels.npy'))
    labels_true = np.load(os.path.join(args.data_path,'moving_mnist_test_with_labels_labels.npz'))
    labels_true = labels_true['arr_0']
    reshaped_data = np.reshape(data,(data.shape[0],-1))

    centers = reshaped_data[0:10]
    polynomials = []
    evaluations = []
    xs = sympy.symbols('x0:%d' % reshaped_data.shape[1])
    centers = xs - centers

    test = reshaped_data[20]

    for i in range(0,10):

        rhs = centers[i]
        rhs = np.array([sympy.Pow(t,2) for t in rhs])

        # TODO change rate of change i.e. modify to be elipsoid and change according to frame number

        temp_rhs = -(args.rate_of_change * (10 - i)) ** 2
        for item in rhs:
            temp_rhs += item
        polynomial = Poly(temp_rhs)
        evaluations.append(polynomial(*test))
        polynomials.append(polynomial)
    if np.all(evaluations==True):
        print('its in')
    else:
        print('its not in')

    solution_area = 1








if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='resae_z1_nlevels3_movingmnist')


    parser.add_argument('--data_path', default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data')
    parser.add_argument('--base_logdir',
                        default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/logs')
    parser.add_argument('--resume', default=False)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    args = parser.parse_args()

    print(args.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.logdir = os.path.join(args.base_logdir, args.name) + '/'


    tf.set_random_seed(0)


    args.clip_gradient = 100.
    args.image_size = 64
    args.rate_of_change = 5


    run_hyperspheres(args)
