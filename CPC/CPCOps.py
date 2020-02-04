import tensorflow
import glob
import os
import numpy as np
import gzip
import scipy
from tensorflow.python.keras.utils.data_utils import Sequence


class MnistHandler(object):

    ''' Provides a convenient interface to manipulate MNIST data '''

    def __init__(self, args):
        self.args = args

        self.X, self.y = self.load_dataset()

    def load_dataset(self):

        if self.args.mode == 'Training':
            data_filename = os.path.join(self.args.data_path, 'moving_mnist_train_with_labels.npz')
            label_filename = os.path.join(self.args.data_path, 'moving_mnist_train_with_labels_labels.npz')
        else:
            data_filename = os.path.join(self.args.data_path, 'moving_mnist_test_with_labels.npz')
            label_filename = os.path.join(self.args.data_path, 'moving_mnist_test_with_labels_labels.npz')

        data = np.load(data_filename)
        data = data['arr_0'].reshape(-1, 1, 64, 64)
        data = data / np.float32(255)
        data = np.array([data[i:i + 30] for i in range(0, len(data), 30)])

        labels = np.load(label_filename)
        labels = labels['arr_0']
        labels = np.array([labels[i:i + 30] for i in range(0, len(labels), 30)])

        return data, labels

    def get_n_samples(self):

        y_len = self.y.shape[0]
        return y_len

    def get_batch_by_labels(self, labels, images_size):
        # Find samples matching labels
        idxs = []
        _labels = labels[0::self.args.terms]
        for i, label in enumerate(_labels):
            idx = np.where(self.y == label)[0]
            idx_sel = np.random.choice(idx, 1)[0]
            idx_to_start = np.random.randint(0, 30-self.args.terms, 1)[0]
            idxs.append((idx_sel, idx_to_start))

        # idxs = np.array(idxs)
        batch = np.array([self.X[i, j:j+self.args.terms, ...] for i, j in idxs]).reshape((-1, 1, images_size, images_size))

        batch = self.process_batch(batch, len(labels), images_size, self.args.rescale)

        return batch

    def get_batch_by_labels_deltas(self, labels, images_size, sentence_labels):
        # Find samples matching labels
        idxs = []
        _labels = labels[0::self.args.terms]
        _sentence_labels = sentence_labels[0::self.args.terms]
        for i, label in enumerate(_labels):
            idx = np.where(self.y == label)[0]
            idx_sel = np.random.choice(idx, 1)[0]
            idx_to_start = np.random.randint(0, 30-self.args.terms, 1)[0]
            idxs.append((idx_sel, idx_to_start))

        # fix the line below to work for non equal terms and negative terms
        batch = np.array([self.X[i, j:j+self.args.terms, ...] for i, j in idxs]).reshape((-1, 1, images_size, images_size))

        batch = self.process_batch(batch, len(labels), images_size, self.args.rescale)

        return batch

    def process_batch(self, batch, batch_size, image_size=64, rescale=True):

        # with the above and the below we are looking at patches as dictated by CPC paper

        batch = batch.reshape((batch_size, 1, image_size, image_size))
        # Rescale to range [-1, +1]
        if rescale:
            batch = batch * 2 - 1

        # Channel last
        batch = batch.transpose((0, 3, 2, 1))

        return batch


class MovingMNISTDataGenerator(Sequence):

    def __init__(self, args, mode='Training'):
        self.args = args
        self.args.mode = mode
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.mnist_handler = MnistHandler(self.args)
        self.n_samples = self.mnist_handler.get_n_samples() // self.args.terms
        self.n_batches = self.n_samples // self.args.batch_size
        self.positive_samples = self.args.batch_size // 2
        self.shape = (self.batch_size, self.mnist_handler.get_n_samples(), self.image_size, self.image_size, 1)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __getitem__(self, item):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        sentence_labels = np.ones((self.args.batch_size, 1)).astype('int32')
        image_labels = np.zeros((self.args.batch_size, self.args.terms + self.args.negative_terms))
        positive_samples_n = self.positive_samples

        for b in range(self.args.batch_size):
            seed = np.mod(np.random.randint(1, 10), 10)  # for some reason randint(0,9) gives disproportionally more 0s
            sentence = np.repeat(seed, (self.args.terms + self.args.negative_terms))
            if positive_samples_n <= 0:

                numbers = np.arange(0, 10)
                predicted_terms = sentence[-self.args.negative_terms:]
                predicted_terms = np.repeat(np.random.choice(numbers[numbers != seed], 1), predicted_terms.shape[0])
                sentence[-self.args.negative_terms:] = predicted_terms
                sentence_labels[b, :] = 0

            # Save sentence
            image_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual images
        images = self.mnist_handler.get_batch_by_labels(image_labels.flatten(), self.args.image_size)

        # Assemble batch
        images = images.reshape((self.args.batch_size, self.args.terms + self.args.negative_terms,
                                 images.shape[1], images.shape[2], images.shape[3]))
        x_images = images[:, :-self.args.negative_terms, ...]
        y_images = images[:, -self.args.negative_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)

        sentence_labels = sentence_labels[idxs, ...]
        # sentence_labels_one_hot = np.zeros((sentence_labels.shape[0],))
        # sentence_labels_one_hot[sentence_labels[:, 0] == 1] = 1
        # sentence_labels_one_hot = np.stack((np.abs(1 - sentence_labels_one_hot), sentence_labels_one_hot), axis=-1)

        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels  # x_images[idxs, ...]



class MovingMNISTDataGenerator_deltas(Sequence):

    def __init__(self, args, mode='Training'):
        self.args = args
        self.args.mode = mode
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.mnist_handler = MnistHandler(self.args)
        self.n_samples = self.mnist_handler.get_n_samples() // self.args.terms
        self.n_batches = self.n_samples // self.args.batch_size
        self.positive_samples = self.args.batch_size // 2
        self.shape = (self.batch_size, self.mnist_handler.get_n_samples(), self.image_size, self.image_size, 1)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __getitem__(self, item):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        sentence_labels = np.ones((self.args.batch_size, self.args.terms + self.args.negative_terms)).astype('int32')
        image_labels = np.zeros((self.args.batch_size, self.args.terms + self.args.negative_terms))

        for b in range(self.args.batch_size):
            seed = np.mod(np.random.randint(1, 10), 10)  # for some reason randint(0,9) gives disproportionally more 0s
            sentence = np.repeat(seed, (self.args.terms + self.args.negative_terms))

            numbers = np.arange(0, 10)
            predicted_terms = sentence[-self.args.negative_terms:]
            predicted_terms = np.repeat(np.random.choice(numbers[numbers != seed], 1), predicted_terms.shape[0])
            sentence[-self.args.negative_terms:] = predicted_terms
            sentence_labels[b, self.args.terms:] = 0

            # Save sentence
            image_labels[b, :] = sentence

        # Retrieve actual images
        images = self.mnist_handler.get_batch_by_labels_deltas(image_labels.flatten(),
                                                               self.args.image_size,
                                                               sentence_labels.flatten())

        # Assemble batch
        images = images.reshape((self.args.batch_size, self.args.terms + self.args.negative_terms,
                                 images.shape[1], images.shape[2], images.shape[3]))
        x_images = images[:, :-self.args.negative_terms, ...]
        x_plus_t_images = x_images[:, 1:, ...]
        x_images = x_images[:, 0, ...]

        y_images = images[:, -self.args.negative_terms:, ...]
        y_images = y_images[:, 1:, ...]

        x_times = np.tile(np.arange(1, self.args.terms),(self.args.batch_size,1))
        y_times = np.tile(np.arange(1, self.args.negative_terms),(self.args.batch_size,1))


        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        x_times = x_times[idxs, ...]
        y_times = y_times[idxs, ...]
        sentence_labels = sentence_labels[idxs, 1:-1, ...]

        image_plus_time = np.concatenate([x_plus_t_images, y_images], axis=1)
        times = np.concatenate([x_times, y_times], axis=1)

        return [x_images[idxs, ...], image_plus_time, times], sentence_labels
