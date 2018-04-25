# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''

import tensorflow as tf

# create Variable objects, 'trainable=True or False' indicates whether variable is trainable or not
val_a = tf.Variable(tf.constant(0.0), name='val_a', dtype=tf.float32)
val_b = tf.Variable(tf.constant(0.0), trainable=False, name='val_b', dtype=tf.float32)
val_c = tf.Variable(tf.constant(0.0), trainable=True, name='val_c', dtype=tf.float32)

# print the whole trainable variables
trainable_variables = tf.trainable_variables()
print('trainable variables')
for tv in trainable_variables:
    print(tv)

# create add operation
add_op = tf.add(val_a, 1)
update_collection = tf.assign(val_a, add_op)

with tf.Session() as sess:
    # initialize variables before using
    tf.global_variables_initializer().run()
    for i in range(5):
        update = sess.run(update_collection)
        print('ite {}:{}'.format(i, update))
