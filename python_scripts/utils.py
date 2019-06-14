"""utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def record_read_and_decode(serialized_example, num_channels, num_classes):
  """function to read record files
  Args:
    serialized_example: .
    num_channels: `int`
    num_classes: `int`
  Returns:
    image: `tensor`, input image
    mask: `tensor`, ground-truth mask
    """
  features = tf.parse_single_example(
      serialized_example,
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'image_raw': tf.FixedLenFeature([], tf.string),
          'mask_raw': tf.FixedLenFeature([], tf.string),
      })
  image = tf.decode_raw(features['image_raw'], tf.float32)
  mask = tf.decode_raw(features['mask_raw'], tf.float32)
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  image = tf.reshape(image, (height, width, num_channels))
  mask = tf.reshape(mask, (height, width, 1))
  mask = tf.cast(mask, tf.int32)
  mask = tf.one_hot(mask, num_classes)
  mask = tf.squeeze(mask, axis=2)
  tf.cast(mask, tf.float32)
  return image, mask
def prepare_dataset(filenames, FLAGS, data_size):
  """function to prepare dataset
  Args:
    filenames: `list`, list of record files.
    FLAGS: `tf.app.flags`.
    data_size: `int`, total number of examples.
  Returns:
    `Tensor` iterator for tf.dataset
    """
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(lambda x: record_read_and_decode(x, FLAGS.num_channels,
                                                         FLAGS.num_classes))
  dataset = dataset.batch(FLAGS.batch_size)
  dataset = dataset.repeat(FLAGS.total_epochs)
  dataset = dataset.shuffle(buffer_size=data_size)
  dataset = dataset.prefetch(1)
  return dataset.make_one_shot_iterator()
