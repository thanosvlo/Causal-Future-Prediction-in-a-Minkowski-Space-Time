import tensorflow as tf
import numpy as np
import os
import keras as k
import glob
from CPCOps import MovingMNISTDataGenerator as data_generator

def dice_coef_loss(y_true, y_pred, smooth=1):
    """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
    intersection = k.backend.sum(k.backend.abs(y_true * y_pred), axis=-1)
    dice_coef = (2. * intersection + smooth) / (k.backend.sum(k.backend.square(y_true), -1) + k.backend.sum(k.backend.square(y_pred), -1) + smooth)
    return 1 - dice_coef

def total_loss(y_true, y_pred):
    class_pred , recon_pred = y_pred
    class_true , recon_true = y_true

    dice_loss = dice_coef_loss(class_true,class_pred)
    recon_loss = k.losses.mean_squared_error(recon_true,recon_pred)

    return dice_loss + recon_loss


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


def decoder_model(x,args):
    # 64x64
    x = k.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    # 32x32
    x = k.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    # 16x16
    # x = k.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    # 8x8
    # x = k.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    # x = k.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    # 4x4
    x = k.layers.Conv2DTranspose(64, 3, strides=1, padding='same', activation='relu')(x)
    # x = k.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(x)
    # 2x2
    x = k.layers.Conv2DTranspose(32, 3, strides=1, padding='same', activation='relu')(x)
    # 1x1
    # x = k.layers.Flatten()(x)
    x = k.layers.Conv2DTranspose(1,  3, strides=1, padding='same', activation='relu',name='recon_output')(x)
    # N x latent
    # x = k.layers.Reshape((args.latent_dim,), name='encoder_embedding')(x)

    return x

def get_model(args):
    k.backend.set_learning_phase(1)

    input_shape = (args.image_size,args.image_size,1)
    input_encoder = k.layers.Input(input_shape)

    encoded = encoded_model(input_encoder, args)
    encoder_model = k.models.Model(input_encoder, encoded, name='encoder')

    # now with the positive and negative terms

    x_input = k.layers.Input((args.terms, input_shape[0], input_shape[1], input_shape[2]))
    x_encoded = k.layers.TimeDistributed(encoder_model)(x_input)

    y_input = k.layers.Input((args.negative_terms, input_shape[0], input_shape[1], input_shape[2]))
    y_encoded = k.layers.TimeDistributed(encoder_model)(y_input)

    #
    dot_product_probs = CPCLayer()([x_encoded, y_encoded])

    # Model
    cpc_model = k.models.Model(inputs=[x_input, y_input], outputs=[dot_product_probs])

    # Compile model
    cpc_model.compile(
        optimizer=k.optimizers.Adam(lr=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return cpc_model


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

    model = get_model(args)

    # Callbacks
    callbacks = [#k.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-4),
                 k.callbacks.ModelCheckpoint(args.logdir + args.name + '.hdf5', monitor='val_loss', verbose=1,
                                                    save_best_only=True)
                 ]

    # Trains the model
    model.fit_generator(
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
    parser.add_argument('--name', default='keras_cpc_2terms')
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--latent_dim', default=100)
    parser.add_argument('--n_levels', default=3)
    parser.add_argument('--terms', default=2)
    parser.add_argument('--negative_terms', default=2)
    parser.add_argument('--rescale', default=False)

    parser.add_argument('--data_path', default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data')
    parser.add_argument('--base_logdir',
                        default='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/logs')
    parser.add_argument('--resume', default=False)
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--train', default=False)
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
