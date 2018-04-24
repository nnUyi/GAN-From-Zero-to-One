# coding='utf-8'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np

from tqdm import tqdm
from ops import *
from utils import get_data, gen_batch_data, save_image

class AutoEncoder:
    model_name = 'AutoEncoder.model'
    def __init__(self, config=None, sess=None):
        self.batchsize = config.batchsize
        # mnist data
        self.input_channel = config.input_channel
        self.output_channel = config.input_channel
        self.input_height = config.input_height
        self.input_width = config.input_width
        
        self.config = config
        self.sess = sess
    
    # encoder
    def encoder(self, input_x, scope_name='encoder', is_training=True, reuse=False):
        # Architecture: (64)5c2s_BR-(128)5c2s_BR-(256)5c2s_BR-(dropout)-FC1024-BR-FC100(laten_code)
        
        # batch normalization parameters setting
        batch_norm_params = {'is_training':is_training, 'decay':0.999, 'epsilon':1e-8, 'updates_collections':None}

        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 weights_regularizer=None,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 activation_fn=tf.nn.relu):
                # [self.batchsize, 28,28,1]
                conv1 = slim.conv2d(input_x, 64, [5,5], stride=2, scope='conv1')
                # [self.batchsize, 14,14,1]
                conv2 = slim.conv2d(conv1, 128, [5,5], stride=2, scope='conv2')
                # [self.batchsize, 7, 7, 1]
                conv3 = slim.conv2d(conv2, 256, [5,5], stride=2, scope='conv3')
                # [self.batchsize, 4, 4, 1]
                flat = tf.reshape(conv3, [self.batchsize, -1])
                # 1024
                fc0 = slim.fully_connected(flat, 1024, scope='fc0')
                # 512
                laten_code = slim.fully_connected(fc0, 100, activation_fn=None, normalizer_fn=None, scope='laten_code')
                
                return laten_code
    
    # decoder
    def decoder(self, laten_code, scope_name='decoder', is_training=True, reuse=False):
        # Architecture: 7*7*128FC_R-(64)5dc2s_BR-(1)5dc2s
        
        # batch normalization parameters setting
        batch_norm_params = {'is_training':is_training, 'decay':0.999, 'epsilon':1e-8, 'updates_collections':None}
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected], 
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 weights_regularizer=None,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 activation_fn=tf.nn.relu):
                # 100 -> 7*7*256
                fc0 = slim.fully_connected(laten_code, 7*7*128, normalizer_fn=None, scope='fc0')
                # [self.batchsize, 7,7,128]
                deconv0 = tf.reshape(fc0, [self.batchsize, 7,7,128])
                # [self.batchsize, 14,14,64]
                deconv1 = slim.conv2d_transpose(deconv0, 64, [5,5], stride=2, scope='deconv1')
                # [self.batchsize, 28,28,1]
                output_x = slim.conv2d_transpose(deconv1, 1, [5,5], stride=2, normalizer_fn=None, activation_fn=None, scope='output_x')
                
                return output_x
                
    def build_model(self):
        # real data
        self.input_x = tf.placeholder(dtype=tf.float32,
                                      shape=[self.batchsize, self.input_height, self.input_width, self.input_channel],
                                      name='input_x')
        # encoder forward training phase
        self.laten_code = self.encoder(self.input_x, is_training=True, reuse=False)
        # decoder forward training phase
        self.recon_x = self.decoder(self.laten_code, is_training=True, reuse=False)
        
        # loss function
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.recon_x, self.input_x)), reduction_indices=[1,2,3])))
                        
        # optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                              beta1=self.config.beta1,
                                              beta2=self.config.beta2).minimize(self.loss)
                                              
        # visualize loss in borwser using tensorboard
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.laten_code_summary = tf.summary.histogram('laten code', self.laten_code)
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
                input_x, input_label = next(data_gen)
                # optimization
                _, loss, recon_x, summaries = self.sess.run([self.optim, self.loss, self.recon_x, self.summaries], feed_dict={self.input_x:input_x})
                # visualize loss in browser using tensorboard
                counter = counter + 1
                self.summary_writer.add_summary(summaries, global_step=counter)
                
            # visualize results of reconstruction
            save_image([8,8], '{}/sampel_epoch_{}.png'.format(self.config.sample_dir, epoch), recon_x)
            
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
