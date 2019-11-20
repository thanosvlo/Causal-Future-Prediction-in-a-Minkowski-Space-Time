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
labels = np.load(labels_file)
labels = labels['arr_0']
train_tfrecord = tf.python_io.TFRecordWriter(os.path.join(out_file,'train_with_labels.tfrecord'))

for i in range(data.shape[0]):
    image = data[i]
    image = np.transpose(image)
    label = labels[i]
    print(i)

    example = tf.train.Example(features=tf.train.Features(feature={
        'data':_bytes_feature(image.tostring()),
        'labels':_bytes_feature(label.tostring())
    }))
    train_tfrecord.write(example.SerializeToString())


train_tfrecord.close()