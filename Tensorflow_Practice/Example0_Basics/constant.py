# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''

import tensorflow as tf

# create a constant Tensor with shape = [1]
tensor_a = tf.constant(0.0)
# create a constant Tensor with shape = [2,2]
tensor_b = tf.constant(0.0, dtype=tf.float32, shape=[2,2])
# create a constant Tensor with shape = [3,3]
tensor_c = tf.constant(0.0, dtype=tf.float32, shape=[3,3], name='tensor_c')

# print Tensor directly
print(tensor_a)
print(tensor_b)
print(tensor_c)

# create a session
with tf.Session() as sess:
    constant_a, constant_b, constant_c = sess.run([tensor_a, tensor_b, tensor_c])
    print(constant_a)
    print(constant_b)
    print(constant_c)
