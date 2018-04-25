# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''

import tensorflow as tf
import numpy as np

# create a placeholder
input_a = tf.placeholder(dtype=tf.float32, shape=[2,3], name='placeholder_a')

# create a constant
tensor_w = tf.constant(1.0, shape=[3,3], dtype=tf.float32, name='tensor_w')
tensor_bias = tf.constant(0.0, shape=[3], dtype=tf.float32, name='tensor_bias')

# create variable
w = tf.Variable(tensor_w, name='w', dtype=tf.float32)
bias = tf.Variable(tensor_bias, name='bias', dtype=tf.float32)

# create multiply operation
multiply_op = tf.add(tf.matmul(input_a, w), bias)

# create initial value of input_a placeholder
init_value = np.random.uniform(-1, 1, [2,3]).astype(np.float32)
print('initial input_a:', init_value)

with tf.Session() as sess:
    # initialize global variables
    tf.global_variables_initializer().run()
    
    update_multiply = sess.run([multiply_op], feed_dict={input_a:init_value})
    print('update_multiply:', update_multiply)

