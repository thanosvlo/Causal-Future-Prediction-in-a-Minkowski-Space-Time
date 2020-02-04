import tensorflow as tf
import os
import numpy as np



def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


in_file = '/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data/moving_mnist_train_with_labels.npz'
labels_file = '/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data/moving_mnist_train_with_labels_labels.npz'
out_file ='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data/'
data = np.load(in_file)
data = data['arr_0']
data = data.transpose((0, 3, 2, 1))
data = np.array([data[i:i + 30] for i in range(0, len(data), 30)])

labels = np.load(labels_file)
labels = labels['arr_0']
labels = np.array([labels[i:i + 30] for i in range(0, len(labels), 30)])

train_tfrecord = tf.python_io.TFRecordWriter(os.path.join(out_file,'train_with_labels_sequence_pos_neg.tfrecord'))

for i in range(2000):#(data.shape[0]-1):
    for j in range(30-7):
        image = data[i][j]
        # image = np.transpose(image)
        label = labels[i][j]
        print(i,j)
        for f in range(5):
            other_sample = data[i][f]
            ys = 1
            times = f+1
            example = tf.train.Example(features=tf.train.Features(feature={
                'data':_bytes_feature(image.tostring()),
                'test_data':_bytes_feature(other_sample.tostring()),
                'test_labels': _int64_feature(ys),
                'times': _int64_feature(times),
                'labels':_int64_feature(label)
            }))
            train_tfrecord.write(example.SerializeToString())
        for f in range(5):
            other_sample = data[i + 1][f]
            ys = 0
            times = f+1
            example = tf.train.Example(features=tf.train.Features(feature={
                'data':_bytes_feature(image.tostring()),
                'test_data':_bytes_feature(other_sample.tostring()),
                'test_labels': _int64_feature(ys),
                'times': _int64_feature(times),
                'labels':_int64_feature(label)
            }))
            train_tfrecord.write(example.SerializeToString())


train_tfrecord.close()