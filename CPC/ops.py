import numpy as np
import tensorflow as tf
import os
import glob


def get_data_moving_MNIST(args):
    device = "/cpu:0"
    with tf.device(
            tf.train.replica_device_setter(0, worker_device=device)):
        def _parse_function(serialized_example):
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'data': tf.FixedLenFeature([], tf.string),
                    'labels': tf.FixedLenFeature([], tf.string),
                })
            image = features['data']
            labels = features['labels']
            image = tf.decode_raw(image, tf.uint8)
            labels = tf.decode_raw(labels, tf.uint8)
            image = tf.reshape(image, [64, 64, 1])
            labels = tf.reshape(labels, [1])
            return image, labels

        # dataset = tf.data.TFRecordDataset(gfile.Glob(FLAGS.data_path+'/*'))
        if args.train:
            print('Train files:', glob.glob(os.path.join(args.data_path , 'train_with_labels.tfrecord')))
            files = tf.data.Dataset.list_files(glob.glob(os.path.join(args.data_path , 'train_with_labels.tfrecord')))
        else:
            print('Test files:', glob.glob(os.path.join(args.data_path, 'test.tfrecord')))
            files = tf.data.Dataset.list_files(glob.glob(os.path.join(args.data_path, 'test.tfrecord')))

        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=100, block_length=20)
        dataset = dataset.map(_parse_function)  # Parse the record into tensors.
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.shuffle(5000)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(1)
        images, labels = dataset.make_one_shot_iterator().get_next()
        return images, labels


def ae_optimization(args, x_in, im_out):
    loss = tf.nn.l2_loss(x_in - im_out)

    momentum = 1. - args.momentum
    decay = 1. - args.decay
    optimizer = tf.train.AdamOptimizer(
        learning_rate=args.learning_rate,
        beta1=momentum, beta2=decay, epsilon=1e-08,
        use_locking=False, name="Adam")

    step = tf.get_variable(
        "global_step", [], tf.int64,
        tf.zeros_initializer(),
        trainable=False)

    grads_and_vars = optimizer.compute_gradients(
        loss, tf.trainable_variables(scope='autoencoder'))
    grads, vars_ = zip(*grads_and_vars)
    capped_grads, gradient_norm = tf.clip_by_global_norm(
        grads, clip_norm=args.clip_gradient)

    capped_grads_and_vars = zip(capped_grads, vars_)
    train_step = optimizer.apply_gradients(
        capped_grads_and_vars, global_step=step)

    return loss, step, train_step ,step


def clf_optimization(args, labels, class_out):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=class_out)
    loss = tf.reduce_mean(loss)
    momentum = 1. - args.momentum
    decay = 1. - args.decay
    optimizer = tf.train.AdamOptimizer(
        learning_rate=args.learning_rate,
        beta1=momentum, beta2=decay, epsilon=1e-08,
        use_locking=False, name="Adam")

    step = tf.get_variable(
        "global_step", [], tf.int64,
        tf.zeros_initializer(),
        trainable=False)

    grads_and_vars = optimizer.compute_gradients(
        loss, tf.trainable_variables(scope='classifier'))
    grads, vars_ = zip(*grads_and_vars)
    capped_grads, gradient_norm = tf.clip_by_global_norm(
        grads, clip_norm=args.clip_gradient)

    capped_grads_and_vars = zip(capped_grads, vars_)
    train_step = optimizer.apply_gradients(
        capped_grads_and_vars, global_step=step)

    return loss, step, train_step ,step



def make_tensorboard(x_in, im_out, loss, args):
    x_in = tf.reshape(x_in, [args.batch_size, args.image_size, args.image_size, 1])
    side_shown = int(np.sqrt(args.batch_size))
    shown_x = tf.transpose(
        tf.reshape(
            x_in[:(side_shown * side_shown), :, :, :],
            [side_shown, args.image_size * side_shown, args.image_size, 1]),
        [0, 2, 1, 3])
    shown_x = tf.transpose(
        tf.reshape(
            shown_x,
            [1, args.image_size * side_shown, args.image_size * side_shown, 1]),
        [0, 2, 1, 3]) * 255.
    tf.summary.image(
        "io/inputs",
        tf.cast(shown_x, tf.uint8),
        max_outputs=1)
    if  args.classifier:
        tf.summary.scalar('out_class', tf.reshape(loss, []))
    else:
        im_out = tf.reshape(im_out, [args.batch_size, args.image_size, args.image_size, 1])
        side_shown = int(np.sqrt(args.batch_size))
        shown_x_sample = tf.transpose(
            tf.reshape(
                im_out[:(side_shown * side_shown), :, :, :],
                [side_shown, args.image_size * side_shown, args.image_size, 1]),
            [0, 2, 1, 3])
        shown_x_sample = tf.transpose(
            tf.reshape(
                shown_x_sample,
                [1, args.image_size * side_shown, args.image_size * side_shown, 1]),
            [0, 2, 1, 3]) * 255.
        tf.summary.image(
            "io/outputs",
            tf.cast(shown_x_sample, tf.uint8),
            max_outputs=1)

    # tf.summary.scalar("loss_flow", tf.reshape(loss_flow_only, []))
    tf.summary.scalar('loss',tf.reshape(loss,[]))
    # tf.summary.scalar("loss_ae", tf.reshape(loss_ae_only, []))
    # tf.summary.scalar('bit per dim',tf.reshape(bit_per_dim,[]))
    summary_op = tf.summary.merge_all()

    return summary_op
