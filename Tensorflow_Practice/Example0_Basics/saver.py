# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/25/2018
github: https://github.com/nnUyi

tensorflow version:1.3.0
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os


data_size = 200
batchsize = 32
model_name = 'MLP'
model_dir = 'checkpoint'

# create model_dir
if not os.path.exists(model_dir):
    os.mkdir(model_dir)    

# generate datasets
data_x = np.linspace(-1, 1, data_size)
data_x = np.expand_dims(data_x, 1)
noise = np.random.normal(0, 0.1, size=[data_size, 1])
data_y = np.power(data_x, 2) + noise

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

# define model saver
saver = tf.train.Saver()

# training phase
with tf.Session() as sess:
    # initialize variables
    tf.global_variables_initializer().run()
    
    # load model
    is_exists = os.path.isfile('{}/{}'.format(model_dir, 'checkpoint'))
    if is_exists:
        saver.restore(sess, '{}/{}'.format(model_dir, model_name))
        print('[***] load model successfully')
    else:
        print('[!!!] fail to load model')

    # training 
    counter = 0
    for epoch in range(30):
        indices = range(data_size)
        np.random.shuffle(indices)
        
        for index in range(data_size/batchsize):
            indice = indices[index*batchsize:(index+1)*batchsize]
            
            train_x = data_x[indice]
            train_y = data_y[indice]
            _, l = sess.run([adam_optim, loss], feed_dict={x:train_x, y:train_y})
            
            counter = counter + 1
            loss_set.append(l)

    # save model
    saver.save(sess, '{}/{}'.format(model_dir, model_name))
