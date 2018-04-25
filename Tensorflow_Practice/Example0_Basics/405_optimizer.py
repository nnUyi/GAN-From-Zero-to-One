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

# define optimizer
sdg_optim = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)
momentum_optim = tf.train.MomentumOptimizer(learning_rate=0.02, momentum=0.9).minimize(loss)
adam_optim = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss)
rmsprop_optim = tf.train.RMSPropOptimizer(learning_rate=0.02).minimize(loss)

optim_sets = [sdg_optim, momentum_optim, adam_optim, rmsprop_optim]
loss_sets = [[],[],[],[]]

# training phase
with tf.Session() as sess:
    # initialize variables
    tf.global_variables_initializer().run()
    
    # training 
    for epoch in range(30):
        indices = range(data_size)
        np.random.shuffle(indices)
        
        for index in range(data_size/batchsize):
            indice = indices[index*batchsize:(index+1)*batchsize]
            
            train_x = data_x[indice]
            train_y = data_y[indice]
            
            for optim, loss_set in zip(optim_sets, loss_sets):
                _, l = sess.run([optim, loss], feed_dict={x:train_x, y:train_y})
                loss_set.append(l)

# plot training loss
labels = ['sdg_optim', 'momentum_optim', 'adam_optim', 'rmsprop_optim']
_, ax = plt.subplots()
for i, l_his in enumerate(loss_sets):
    ax.plot(l_his, label=labels[i])

ax.set_title('loss comparision')
ax.set_xlabel('steps')
ax.set_ylabel('loss')
ax.legend()
plt.show()
