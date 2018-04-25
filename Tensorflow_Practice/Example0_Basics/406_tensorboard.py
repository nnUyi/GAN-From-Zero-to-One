# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

data_size = 200
batchsize = 32

# generate datasets
data_x = np.linspace(-1, 1, data_size)
data_x = np.expand_dims(data_x, 1)
noise = np.random.normal(0, 0.1, size=[data_size, 1])
data_y = np.power(data_x, 2) + noise

# plot datasets
_, ax = plt.subplots()
ax.scatter(data_x, data_y, s=50, color='g', alpha=0.5, label='datasets')
ax.set_title('data plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()

# build model
x = tf.placeholder(dtype=tf.float32, shape=[batchsize, 1], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[batchsize, 1], name='y')

# 1->20->1 MLP neural networkss
hidden_layer = slim.fully_connected(x, 20,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    scope='hidden_layer')
output_layer = slim.fully_connected(hidden_layer, 1, 
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    activation_fn=None,
                                    scope='output_layer')

# define loss function
loss = tf.reduce_mean(tf.square(output_layer-y))
loss_set = []

# define optimizer
adam_optim = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss)

# training phase
with tf.Session() as sess:
    # define summary
    # define summary scalar
    loss_summary = tf.summary.scalar('loss', loss)
    # merge summary
    loss_summaries = tf.summary.merge([loss_summary])
    # define summary writer
    summaries_writer = tf.summary.FileWriter('./logs', sess.graph)

    # initialize variables
    tf.global_variables_initializer().run()
    
    # training 
    counter = 0
    for epoch in range(30):
        indices = range(data_size)
        np.random.shuffle(indices)
        
        for index in range(data_size/batchsize):
            indice = indices[index*batchsize:(index+1)*batchsize]
            
            train_x = data_x[indice]
            train_y = data_y[indice]
            _, l, l_s = sess.run([adam_optim, loss, loss_summaries], feed_dict={x:train_x, y:train_y})
            
            counter = counter + 1
            # add summary to logs
            summaries_writer.add_summary(l_s, global_step=counter)
            loss_set.append(l)

'''
After training phase you will find that the logs directory is in your root directory.
You can see some files are stored in logs directory.

Then you need to do the following instructions to see the logs:
1. open your terminal and enter your root directory
2. type the following instructions:
    tensorboard --logdir=logs
3. open your browser and type the ip: 127.0.1.1:6006
4. congratulation to you
'''
