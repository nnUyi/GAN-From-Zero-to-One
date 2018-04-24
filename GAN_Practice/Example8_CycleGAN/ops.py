import tensorflow as tf
import numpy as np
import scipy.misc
import math
import tensorflow.contrib.slim as slim

# instance norm
def instance_norm(input_data_x, name='instance_norm'):
    with tf.variable_scope(name):
        depth = input_data_x.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_data_x, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_data_x-mean)*inv
        return scale*normalized + offset

# residual block
def res_block(input_x, output_dim, kernel_size=3, stride=1, name='res'):
    p = int((kernel_size-1)/2)
    x_padd = tf.pad(input_x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x_padd = tf.nn.relu(instance_norm(conv2d(x_padd, output_dim ,kernel_size, stride, scope_name=name+'_c1', conv_type='VALID'), name=name+'_in1'))
    x_padd = tf.pad(x_padd, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    x_padd = tf.nn.relu(instance_norm(conv2d(x_padd, output_dim, kernel_size, stride, scope_name=name+'_c2', conv_type='VALID'), name=name+'_in2'))
    return input_x+x_padd
    
# convolution
def conv2d(input_x, output_dim, kernel_size=3, stride=2,stddev=0.02, scope_name='conv2d', conv_type='SAME'):
    with tf.variable_scope(scope_name):
        return slim.conv2d(input_x, output_dim, kernel_size, stride, padding=conv_type, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

# deconv2d
def deconv2d(input_x, output_dim, kernel_size = 3, stride=2, stddev=0.02, scope_name="deconv2d", conv_type='SAME'):
    with tf.variable_scope(scope_name):
        return slim.conv2d_transpose(input_x, output_dim, kernel_size, stride, padding=conv_type, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

# batch normalization
def batch_norm(input_x, epsilon=1e-5, momentum=0.9, is_training = True, name='batch_name'):
    with tf.variable_scope(name) as scope:
        batch_normalization = tf.contrib.layers.batch_norm(input_x,
                                              decay=momentum,
                                              updates_collections=None,
                                              epsilon=epsilon,
                                              scale=True,
                                              is_training=is_training,
                                              scope=name)
        return batch_normalization
        
# fully connected
def linear(input_x, output_size, scope_name='linear'):
    shape = input_x.get_shape()
    input_size = shape[1]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', [input_size, output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))        
        output = tf.matmul(input_x, weights) + bias
        return output

# leaky_relu
def leaky_relu(input_x, leaky=0.2):
    return tf.maximum(leaky*input_x, input_x)

# pooling
def max_pool(input_data_x, filter_shape=[1,2,2,1], pooling_type='SAME'):
    if pooling_type == 'SAME':
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)
    else:
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)
