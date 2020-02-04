import tensorflow as tf


class ResNetAE4(object):

    def __init__(self,
                 n_ResidualBlock=4,
                 n_levels=3,
                 z_dim=3,
                 output_channels=1,
                 is_trainable=True):

        self.is_trainable = is_trainable
        self.n_ResidualBlock = n_ResidualBlock
        self.n_levels = n_levels
        self.max_filters = 2 ** (n_levels+3)
        self.z_dim = z_dim
        self.output_channels = output_channels

    def ResidualBlock(self, x, filters=64, kernel_size=(3,3), strides=(1,1) , trainable=True):
        """self.
        Full pre-activation ResNet Residual block
        https://arxiv.org/pdf/1603.05027.pdf
        """
        self.is_trainable = trainable
        skip = x
        x = tf.layers.batch_normalization(x, trainable=self.is_trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters,
                             strides=strides, padding='same', use_bias=False,trainable=self.is_trainable)
        x = tf.layers.batch_normalization(x, trainable=self.is_trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters,
                             strides=strides, padding='same', use_bias=False,trainable=self.is_trainable)
        x = x + skip
        return x

    def encoder(self, x, trainable=True):
        """
        'Striving for simplicity: The all convolutional net'
        arXiv: https://arxiv.org/pdf/1412.6806.pdf
        'We find that max-pooling can simply be replaced by a convolutional layer
        with increased stride without loss in accuracy on several image recognition benchmarks'
        """
        self.is_trainable = trainable

        x = tf.layers.conv2d(inputs=x, filters=8,
                             kernel_size=(3,3), strides=(1,1),
                             padding='same', activation=tf.nn.relu)

        skips = []

        for i in range(self.n_levels):

            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (self.n_levels - i)

            skips.append(tf.layers.conv2d(inputs=x, filters=self.max_filters,
                                          kernel_size=(ks, ks), strides=(ks, ks),
                                          padding='same', activation=tf.nn.relu,trainable=self.is_trainable))

            for _ in range(self.n_ResidualBlock):
                x = self.ResidualBlock(x, filters=n_filters_1,trainable=self.is_trainable)

            x = tf.layers.conv2d(inputs=x, filters=n_filters_2,
                                 kernel_size=(2, 2), strides=(2, 2),
                                 padding='same', activation=tf.nn.relu,trainable=self.is_trainable)

        x = tf.add_n([x] + skips)

        x = tf.layers.conv2d(inputs=x, filters=self.z_dim,
                             kernel_size=(3,3), strides=(1,1),
                             padding='same', activation=tf.nn.relu,trainable=self.is_trainable)

        return x

    def decoder(self, z, trainable=True):
        self.is_trainable = trainable

        z = z_top = tf.layers.conv2d(inputs=z, filters=self.max_filters,
                                     kernel_size=(3,3), strides=(1,1),
                                     padding='same', activation=tf.nn.relu,trainable=self.is_trainable)

        for i in range(self.n_levels):

            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i+1)

            z = tf.layers.conv2d_transpose(inputs=z, filters=n_filters,
                                           kernel_size=(2, 2), strides=(2, 2),
                                           padding='same', activation=tf.nn.relu,trainable=self.is_trainable)

            for _ in range(self.n_ResidualBlock):
                z = self.ResidualBlock(z, filters=n_filters,trainable=self.is_trainable)

            z += tf.layers.conv2d_transpose(inputs=z_top, filters=n_filters,
                                            kernel_size=(ks, ks), strides=(ks, ks),
                                            padding='same', activation=tf.nn.relu,trainable=self.is_trainable)

        z = tf.layers.conv2d(inputs=z, filters=self.output_channels,
                             kernel_size=(3,3), strides=(1,1),
                             padding='same', activation=tf.nn.relu,trainable=self.is_trainable)

        return z




class ResNetAE4_class(object):

    def __init__(self,
                 num_of_classes =10,
                 n_ResidualBlock=4,
                 n_levels=3,
                 z_dim=3,
                 output_channels=1,
                 is_trainable=True):

        self.num_of_classes = num_of_classes
        self.is_trainable = is_trainable
        self.n_ResidualBlock = n_ResidualBlock
        self.n_levels = n_levels
        self.max_filters = 2 ** (n_levels+3)
        self.z_dim = z_dim
        self.output_channels = output_channels

    def ResidualBlock(self, x, filters=64, kernel_size=(3,3), strides=(1,1) , trainable=True):
        """self.
        Full pre-activation ResNet Residual block
        https://arxiv.org/pdf/1603.05027.pdf
        """
        self.is_trainable = trainable
        skip = x
        x = tf.layers.batch_normalization(x, trainable=self.is_trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters,
                             strides=strides, padding='same', use_bias=False,trainable=self.is_trainable)
        x = tf.layers.batch_normalization(x, trainable=self.is_trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters,
                             strides=strides, padding='same', use_bias=False,trainable=self.is_trainable)
        x = x + skip
        return x

    def encoder(self, x, trainable=True):
        """
        'Striving for simplicity: The all convolutional net'
        arXiv: https://arxiv.org/pdf/1412.6806.pdf
        'We find that max-pooling can simply be replaced by a convolutional layer
        with increased stride without loss in accuracy on several image recognition benchmarks'
        """
        self.is_trainable = trainable

        x = tf.layers.conv2d(inputs=x, filters=8,
                             kernel_size=(3,3), strides=(1,1),
                             padding='same', activation=tf.nn.relu)

        skips = []

        for i in range(self.n_levels):

            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (self.n_levels - i)

            skips.append(tf.layers.conv2d(inputs=x, filters=self.max_filters,
                                          kernel_size=(ks, ks), strides=(ks, ks),
                                          padding='same', activation=tf.nn.relu,trainable=self.is_trainable))

            for _ in range(self.n_ResidualBlock):
                x = self.ResidualBlock(x, filters=n_filters_1,trainable=self.is_trainable)

            x = tf.layers.conv2d(inputs=x, filters=n_filters_2,
                                 kernel_size=(2, 2), strides=(2, 2),
                                 padding='same', activation=tf.nn.relu,trainable=self.is_trainable)

        x = tf.add_n([x] + skips)

        x = tf.layers.conv2d(inputs=x, filters=self.z_dim,
                             kernel_size=(3,3), strides=(1,1),
                             padding='same', activation=tf.nn.relu,trainable=self.is_trainable)
        x = tf.layers.flatten(inputs=x)
        x = tf.layers.dense(inputs=x,units=self.num_of_classes,activation='softmax')

        return x


class DeltaModel(object):

    def __init__(self,args):
        self.args = args

    def encode(self,x_in):
        x = tf.layers.flatten(x_in)
        # x = tf.layers.conv2d(inputs=x_in, filters=32, kernel_size=3, strides=1,
        #                      padding='same', activation='relu')
        # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, strides=1,
        #                      padding='same', activation='relu')
        # x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, strides=1,
        #                      padding='same', activation='relu')
        # x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, strides=1,
        #                      padding='same', activation='relu')
        # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, strides=1,
        #                      padding='same', activation='relu')
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.dense(inputs=x, units=32, activation='relu')
        x = tf.layers.dense(inputs=x, units=64, activation='relu')

        x = tf.reshape(x, (-1, 8, 8, 1))

        return x
