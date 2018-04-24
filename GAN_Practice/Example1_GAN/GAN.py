# coding='utf-8'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np

from tqdm import tqdm
from ops import *
from utils import get_data, gen_batch_data, save_image

class GAN:
    model_name = 'GAN.model'
    def __init__(self, config=None, sess=None):
        self.z_dim = 100
        self.g_dim = 64
        self.d_dim = 64
        self.batchsize = config.batchsize
        # mnist data
        self.input_channel = config.input_channel
        self.output_channel = config.input_channel
        self.input_height = config.input_height
        self.input_width = config.input_width
        
        self.config = config
        self.sess = sess
    
    def generator(self, z, scope_name='generator', is_training=True, reuse=False):
        # Architecture: FC1024_BR-FC7*7*128_BR-FC14*14*64_BR-FC28*28*1_S
        
        # batch normalization parameters
        batch_norm_params = {'is_training':is_training, 'decay':0.999, 'epsilon':1e-8, 'updates_collections':None}
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected,
                                 slim.conv2d_transpose],
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 weights_regularizer=None,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 activation_fn=tf.nn.relu
                                 ):
                # 100->1024
                fc0 = slim.fully_connected(z, 1024, scope='g_fc0')
                # size = [batchsize, 7, 7, self.g_dim*2]
                fc1 = slim.fully_connected(fc0, 7*7*self.g_dim*2, scope='g_fc1')
                # size = [batchsize, 14, 14, self.g_dim]
                fc2 = slim.fully_connected(fc1, 14*14*self.g_dim, scope='g_fc2')
                # size = [batchsize, 28, 28, 1]
                fc3 = slim.fully_connected(fc2, 28*28*self.output_channel, normalizer_fn=None,
                                                                           activation_fn=None,
                                                                           scope='g_fc3')
                
                logits = tf.reshape(fc3, [self.batchsize, 28, 28, 1])
                return logits, tf.nn.sigmoid(logits)

    def discriminator(self, input_x, scope_name='discriminator', is_training=True, reuse=False):
        # Architecture: FC2048_LR-FC1024_LR-FC512_LR-(dropout)-FC1_S
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected],
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 weights_regularizer=None,
                                 normalizer_fn=None,
                                 activation_fn=None):
                # [batchsize, 28*28*1]
                flat = tf.reshape(input_x, [self.batchsize, -1])
                # 2048
                fc0 = l_relu(slim.fully_connected(flat, 2048, scope='d_fc0'))
                # 1024
                fc1 = l_relu(slim.fully_connected(fc0, 1024, scope='d_fc1'))
                # 512
                fc2 = l_relu(slim.fully_connected(fc1, 512, scope='d_fc2'))
                # fc2_drop = slim.dropout(fc2, 0.5, scope='fc2_drop')
                # 1
                logits = slim.fully_connected(fc2, 1, normalizer_fn=None, scope='d_output')
                return logits
    
    def build_model(self):
        # noise_z
        self.z = tf.placeholder(dtype=tf.float32,
                                shape=[self.batchsize, self.z_dim],
                                name='noise_z')
        # real data
        self.input_x = tf.placeholder(dtype=tf.float32,
                                      shape=[self.batchsize, self.input_height, self.input_width, self.input_channel],
                                      name='input_x')
                                      
        _, self.fake_image = self.generator(self.z, is_training=True, reuse=False)
        _, self.sample = self.generator(self.z, is_training=False, reuse=True)
        
        self.real_unit = self.discriminator(self.input_x, is_training=True, reuse=False)
        self.fake_unit = self.discriminator(self.fake_image, is_training=True, reuse=True)
        
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
                
        self.d_fake_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_unit, tf.zeros_like(self.fake_unit)))
        self.d_real_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.real_unit, tf.ones_like(self.real_unit)))
        # generator loss
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_unit, tf.ones_like(self.fake_unit)))
        # discriminator loss
        self.d_loss = self.d_fake_loss + self.d_real_loss
        
        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        for var in self.d_vars:
            print(var)
        # optimizer
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                              beta1=self.config.beta1,
                                              beta2=self.config.beta2).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                              beta1=self.config.beta1,
                                              beta2=self.config.beta2).minimize(self.g_loss, var_list=self.g_vars)
        # visualize loss in borwser using tensorboard
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.noise_z_summary = tf.summary.histogram('noise', self.z)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('{}/{}'.format(self.config.log_dir, self.config.dataset), self.sess.graph)
        
        # model saver
        self.saver = tf.train.Saver()
        
    def train_model(self):
        # initialize variables
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
            
        # load model if model is exist
        bool_load = self.load_model()
        if bool_load:
            print('[***] load model successfully')
        else:
            print('[!!!] fail to load model')
        
        # get mnist dataset
        datasource = get_data(data_type=self.config.dataset, is_training=self.config.is_training)
        data_gen = gen_batch_data(batchsize=self.batchsize, datasource=datasource)
        
        counter = 0
        for epoch in range(self.config.epoch):
            # save model per 10 epoches
            print('epoch:{}'.format(epoch))
            if np.mod(epoch, 10) == 0:
                self.save_model()

            for ite in tqdm(range(50000/self.batchsize)):
                input_x, _ = next(data_gen)
                noise_z = np.random.uniform(-1,1, size=[self.batchsize, self.z_dim]).astype(np.float)
                
                # optimize discriminator
                _, d_loss, summaries = self.sess.run([self.d_optim, self.d_loss, self.summaries], feed_dict={self.z:noise_z,
                                                                                                             self.input_x:input_x})
                # optimize generator
                _, g_loss = self.sess.run([self.g_optim, self.g_loss], feed_dict={self.z:noise_z})
                # optimize generator
                _, g_loss = self.sess.run([self.g_optim, self.g_loss], feed_dict={self.z:noise_z})

                # sample image during training phase        
                if np.mod(ite, 100)==0:
                    sample = self.sample.eval({self.z:noise_z})
                    save_image([8,8], '{}/sample_{:3d}_{:4d}.png'.format(self.config.sample_dir, epoch, ite), sample)
                    # save_image([8,8], '{}/input_{:3d}_{:4d}.png'.format(self.config.sample_dir, epoch, ite), input_x)
                
                # visualize loss in browser using tensorboard
                counter = counter + 1
                self.summary_writer.add_summary(summaries, global_step=counter)

    @property
    def model_dir(self):
        return '{}/{}'.format(self.config.checkpoint_dir, self.config.dataset)
    @property
    def model_pos(self):
        return '{}/{}/{}'.format(self.config.checkpoint_dir, self.config.dataset, self.model_name)    
        
    def load_model(self):
        if not os.path.exists(self.model_dir):
            return False
        self.saver.restore(self.sess, self.model_pos)
        return True
    
    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.saver.save(self.sess, self.model_pos)

    def test_model(self):
        pass
