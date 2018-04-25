# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''

import tensorflow as tf

is_gpu = True

tensor_a = tf.constant(0.0, shape=[1], dtype=tf.float32, name='tensor_a')
val_a = tf.Variable(tensor_a, dtype=tf.float32, name='val_a')

add_op = tf.add(val_a, 2.0)
update_collection = tf.assign(val_a, add_op)

# config gpu option
# allocate gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33, allow_growth=True)

# allow_soft_placement: once the chosen device is not exists, 
#        setting allow_soft_placement=True allow tensorflow to allocate the avariable device automatically

# logs_device_placement: allow tensorflow to print the logs of the device
if is_gpu:
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)
else:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    update = sess.run(update_collection)
    print(update)
