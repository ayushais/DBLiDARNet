"""different layer implementation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf



def conv_2d(bottom, output, k_size, stride, pad, name):
  """Function for 2D convolution

  Args:
    bottom: `Tensor`, input tensor of shape NxHxWxC.
    output: `int`, number of feature maps.
    k_size: `int`, filter kernel size.
    stride: `int`, stride.
    pad: `string`, padding choice.
    name: `string`, name of the operation.

  Returns:
    `Tensor` of shape NxHxWxoutput.
    """
  with tf.variable_scope(name):
    w_conv = tf.get_variable("w_conv", shape=[k_size, k_size,
                                              bottom.shape.as_list()[3],
                                              output],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(bottom, w_conv, strides=[1, stride, stride, 1],
                        padding=pad)
    b_conv = tf.Variable(tf.constant(0.1, shape=[output]), name="bias")
    return tf.nn.bias_add(conv, b_conv, name="conv")
def transpose_conv_2d(bottom, output, k_size, stride, pad, name):
  """Function for 2D transpose convolution

  Args:
    bottom: `Tensor`, input tensor of shape NxHxWxC.
    output: `int`, number of feature maps.
    k_size: `int`, filter kernel size.
    stride: `int`, stride.
    pad: `string`, padding choice.
    name: `string`, name of the operation.

  Returns:
    `Tensor` of shape NxHxWxoutput.
    """
  with tf.variable_scope(name):
    w_conv = tf.get_variable("w_conv", shape=[k_size, k_size,
                                              bottom.shape.as_list()[3],
                                              output],
                             initializer=tf.contrib.layers.xavier_initializer())
    b_conv = tf.Variable(tf.constant(0.1, shape=[output]), name="bias")
    shape = tf.shape(bottom)
    conv_transpose = tf.nn.conv2d_transpose(bottom, w_conv,
                                            output_shape=[shape[0],
                                                          stride * bottom.shape.as_list()[1],
                                                          stride * bottom.shape.as_list()[2],
                                                          output],
                                            strides=[1, stride, stride, 1],
                                            padding="SAME")
    return tf.nn.bias_add(conv_transpose, b_conv, name="upconv")
def conv_2d_depth_separable(bottom, output, k_size, stride, pad, name,
                            channel_multiplier=1):
  """Function for depth separable 2D convolution

  Args:
    bottom: `Tensor`, input tensor of shape NxHxWxC.
    output: `int`, number of feature maps.
    k_size: `int`, filter kernel size.
    stride: `int`, stride.
    pad: `string`, padding choice.
    name: `string`, name of the operation.
    channel_multiplier: `int`, number of features learned
    by depthwise convoltuion

  Returns:
    `Tensor` of shape NxHxWxoutput.
    """

  with tf.variable_scope(name):
    w_conv_depthwise = tf.get_variable("depth_w_conv",
                                       shape=[k_size, k_size,
                                              bottom.shape.as_list()[3],
                                              channel_multiplier],
                                       initializer=tf.contrib.layers.xavier_initializer())
    w_conv_pointwise = tf.get_variable("point_w_conv",
                                       shape=[1, 1,
                                              bottom.shape.as_list()[3] * \
                                              channel_multiplier,
                                              output],
                                       initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.separable_conv2d(bottom, w_conv_depthwise, w_conv_pointwise,
                                  strides=[1, stride, stride, 1],
                                  padding=pad)
    b_conv = tf.Variable(tf.constant(0.1, shape=[output]), name="bias")
    return tf.nn.bias_add(conv, b_conv, name="conv")
def add_layer(bottom, output, k_size, name, is_training,
              keep_prob=1.0, depth_separable=False):
  """Function to add a layer in a denseblock

  Args:
    bottom: `Tensor`, input tensor of shape NxHxWxC.
    output: `int`, number of feature maps.
    k_size: `int`, filter kernel size.
    name: `string`, name of the operation.
    is_training: `bool`
    keep_prob: `float`, keep probability for dropout
    depth_separable: `bool`
  Returns:
    `Tensor` of shape NxHxWxoutput.
    """
  current_layer = tf.contrib.layers.batch_norm(bottom, scale=True,
                                               is_training=is_training,
                                               updates_collections=None)
  current_layer = tf.nn.relu(current_layer)
  if depth_separable:
    current_layer = conv_2d_depth_separable(current_layer, output,
                                            k_size, 1, 'SAME', name)
    current_layer = tf.nn.dropout(current_layer, rate=1-keep_prob)
  else:
    current_layer = conv_2d(current_layer, output, k_size, 1,
                            'SAME', name)
    current_layer = tf.nn.dropout(current_layer, rate=1-keep_prob)
  return current_layer
def add_block(scope_name, bottom, num_layers, in_features, k_size, growth,
              is_training, keep_prob=1.0,
              depth_separable=False):
  """Function to add a denseblock

  Args:
    scope_name: `string`
    bottom: `Tensor`, input tensor of shape NxHxWxC.
    num_layers: `int`, number of layers in the block.
    in_feature: `int`, number of input feature maps.
    k_size: `int`, filter kernel size.
    growth: `int`, growth rate.
    is_training: `bool`
    keep_prob: `float`, keep probability for dropout
    depth_separable: `bool`
  Returns:
    stack: `Tensor`, tensor of shape NxHxWx(num_layers * growth + in_features)
    features: `int`, total_features
    db_output: Tensor`, tensor of shape NxHxWx(num_layers * growth)
    """
  features = in_features
  stack = bottom
  db_output = []
  db_output = np.array(db_output)
  with tf.variable_scope(scope_name) as scope:
    for idx in range(num_layers):
      current_layer = add_layer(stack, growth, k_size,
                                scope_name+ str(idx) + 'W',
                                is_training, keep_prob, depth_separable)
      if db_output.shape[0] == 0:
        db_output = current_layer
      else:
        db_output = tf.concat([db_output, current_layer], axis=3)
      stack = tf.concat([stack, current_layer], axis=3)
      features += growth
  return stack, features, db_output
