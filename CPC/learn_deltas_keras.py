import tensorflow as tf
import numpy as np
import os
import keras as k
import glob
from CPCOps import MovingMNISTDataGenerator_deltas as data_generator


class InlusioTestLayer(k.layers.Layer):
    ''' Performs Inclusion test
        Inputs: center, slack, times, test points
        Outputs: 0 if its out , 1 if its in
        '''

    def __init__(self, **kwargs):
        super(InlusioTestLayer, self).__init__(**kwargs)

    def call(self, inputs):
        center = inputs[..., 0]
        test_points = inputs[..., 1]
        deltas = inputs[..., 2]
        times = inputs[..., 3]
        dif = k.backend.abs(test_points - center)
        predicted_deltas = deltas * times
        incl = k.backend.greater_equal(predicted_deltas, dif)  # x>y, deltas >= dif --> its in the sphere

        return incl

    def compute_output_shape(self, batch_size):
        return (batch_size, 4)


def inclusiontest(inputs):

    center, test_points, deltas, times = inputs
    dif = k.backend.abs(test_points - center)
    predicted_deltas = deltas * times
    incl = k.backend.greater_equal(predicted_deltas, dif) # x>y, deltas >= dif --> its in the sphere

    return incl


def compute_output_shape(batch_size):
        return (batch_size, 4)

class CPCLayer(k.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, input):

        # Compute dot product among vectors
        preds, y_encoded = input
        dot_product = k.backend.mean(y_encoded * preds, axis=-1)
        dot_product = k.backend.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = k.backend.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, batch_size):
        return (batch_size, 1)

def encoded_model(x,args):
    # 64x64
    x = k.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(units=256, activation='linear')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Dense(units=args.latent_dim, activation='linear', name='encoder_embedding')(x)

    return x


def get_model(args):

    input_shape = (args.image_size, args.image_size,1)
    input_encoder = k.layers.Input(input_shape)

    encoded = encoded_model(input_encoder, args)
    encoder_model = k.models.Model(input_encoder, encoded, name='encoder_model')

    # now with the positive and negative terms

    x_input = k.layers.Input((args.terms, input_shape[0], input_shape[1], input_shape[2]))
    x_encoded = k.layers.TimeDistributed(encoder_model)(x_input)

    y_input = k.layers.Input((args.negative_terms, input_shape[0], input_shape[1], input_shape[2]))
    y_encoded = k.layers.TimeDistributed(encoder_model)(y_input)

    dot_product_probs = CPCLayer()([x_encoded, y_encoded])

    # Model
    cpc_model = k.models.Model(inputs=[x_input, y_input], outputs=[dot_product_probs])
    #
    cpc_model.compile(
        optimizer=k.optimizers.Adam(lr=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return cpc_model


def get_delta_model(args):

    k.backend.set_learning_phase(1)

    # INPUTS
    # frame to predict the deltas for
    x_input = k.layers.Input((args.image_size, args.image_size, 1), name='x_input')

    # frame to do tests with
    y_input = k.layers.Input((np.max((args.terms+args.negative_terms-2, 1)),
                              args.image_size, args.image_size, 1), name='y_input')

    # Times ahead
    input_time_ahead = k.layers.Input((np.max((args.terms+args.negative_terms-2, 1)), 1), name='times')

    # Input to the Delta model
    delta_x_input = k.layers.Input((args.latent_dim,), name='delta_input')

    # Load CPC Model
    model = get_model(args)
    model.load_weights(args.logdir + args.name + '.hdf5')
    encoder_model = model.get_layer("time_distributed_1").layer

    # Freeze weights of original model
    for l in model.layers:
        l.trainable = False
    for l in encoder_model.layers:
        l.trainable = False
    # Delta base model

    x = k.layers.Dense(128, activation='relu')(delta_x_input)
    x = k.layers.Dense(64,activation='relu')(x)
    x = k.layers.Dense(args.latent_dim,activation='relu')(x)
    delta_model = k.models.Model(delta_x_input, x, name='delta_predictor')

    x_encoded = encoder_model(x_input)
    y_encoded = k.layers.TimeDistributed(encoder_model)(y_input)
    predicted_deltas = delta_model(x_encoded)

    tiled_x_encoded = k.backend.tile(x_encoded[:, np.newaxis, ...],
                                     (1, np.max((args.terms+args.negative_terms-2, 1)), 1))
    tiled_predicted_deltas = k.backend.tile(predicted_deltas[:, np.newaxis, ...],
                                            (1, np.max((args.terms + args.negative_terms - 2, 1)), 1))
    tiled_times = k.backend.tile(input_time_ahead, (1, 1, 100))

    stacked_tensors = k.backend.stack([tiled_x_encoded, y_encoded, tiled_predicted_deltas, tiled_times], axis= -1 )
    incl = k.layers.TimeDistributed(InlusioTestLayer())([stacked_tensors])

    delta_models = k.models.Model([x_input, y_input, input_time_ahead], incl)

    delta_models.compile(
        optimizer=k.optimizers.Adam(lr=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return delta_models


def run_inference(args):
    # train_data = data_generator(args,mode='Training')

    test_data = data_generator(args,mode='Testing')

    model = get_model(args)

    # Callbacks
    callbacks = [#k.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-4),
                 k.callbacks.ModelCheckpoint(args.logdir + args.name + '.hdf5', monitor='val_loss', verbose=1,
                                                    save_best_only=True)
                 ]
    model.load_weights(args.logdir + args.name + '.hdf5')

    # Trains the model
    predictions = model.predict_generator(
        generator=test_data,
        steps=len(test_data),
        verbose=1
    )
    evals = model.evaluate_generator(
        generator=test_data,
        steps=len(test_data),
        verbose=1
    )
    print(evals)


def run_train(args):

    train_data = data_generator(args,mode='Training')
    # a = train_data.next()
    test_data = data_generator(args,mode='Testing')

    delta_model = get_delta_model(args)

    # Callbacks
    callbacks = [  # k.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-4),
        k.callbacks.ModelCheckpoint(args.logdir + args.name + '.hdf5', monitor='val_loss', verbose=1,
                                    save_best_only=True)
    ]

    # Trains the model
    delta_model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=len(test_data),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dev')
    parser.add_argument('--transfer_name', default='keras_cpc_2terms')
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--latent_dim', default=100)
    parser.add_argument('--n_levels', default=3)
    parser.add_argument('--terms', default=3)
    parser.add_argument('--negative_terms', default=3)
    parser.add_argument('--rescale', default=False)

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
    args.learning_rate = 1e-3
    args.clip_gradient = 100.
    args.image_size = 64

    if args.train:
        run_train(args=args)
    else:
        # pass
        run_inference(args=args)
