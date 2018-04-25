# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''

import tensorflow as tf

# 1.create variable objects, get_variable is used for **sharing variables**
# 2.sharing variables
# ps: when using tf.get_variable() API,
#            variable name must be unique since the same name of variables will encounter error
#            while using tf.Variable() API, tensorflow allows the same name in different variables,
#            since tensorflow will allocate different IDs to each variable automatically. 
val_a = tf.get_variable(name='val_a',
                        shape=[1],
                        dtype=tf.float32)

val_b = tf.get_variable(name='val_b',
                        shape=[1],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.02))

val_c = tf.get_variable(name='val_c',
                        shape=[1],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.02),
                        trainable=True)

val_d = tf.get_variable(name='val_d',
                        shape=[1],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.02),
                        trainable=False)

# create add operation
add_op = tf.add(val_a, 1)
update_collection = tf.assign(val_a, add_op)

with tf.Session() as sess:
    # initialize variables before using
    tf.global_variables_initializer().run()
    # print initial value of val_a
    init_val_a = sess.run(val_a)
    print(init_val_a)
    
    for i in range(5):
        update = sess.run(update_collection)
        print('ite {}:{}'.format(i, update))
