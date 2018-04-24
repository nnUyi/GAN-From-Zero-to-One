# coding='utf-8'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np

from tqdm import tqdm
from ops import *
from utils import get_data, gen_batch_data, save_image

class FCN:
    model_name = 'FCN.model'
    def __init__(self, config=None, sess=None):
        self.batchsize = config.batchsize
        # mnist data
        self.num_class = config.num_class
        self.input_channel = config.input_channel
        self.output_channel = config.input_channel
        self.input_height = config.input_height
        self.input_width = config.input_width
        
        self.config = config
        self.sess = sess
    
    def fully_connected_network(self, input_x, scope_name='fully_connected_network', is_training=True, reuse=False):
        # Architecture: FC2048_LR-FC1024_LR-FC512_LR-(dropout)-FC10_SoftMax
        # correct rate: 98.69%
        
        # batch normalization parameters setting
        batch_norm_params = {'is_training':is_training, 'decay':0.999, 'epsilon':1e-8, 'updates_collections':None}
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected],
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 weights_regularizer=None,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 activation_fn=tf.nn.relu):
                # [batchsize, 28*28*1]
                flat = tf.reshape(input_x, [self.batchsize, -1])
                # 2048
                fc0 = slim.fully_connected(flat, 2048, scope='fc0')
                # 1024
                fc1 = slim.fully_connected(fc0, 1024, scope='fc1')
                # 512
                fc2 = slim.fully_connected(fc1, 512, scope='fc2')
                # fc2_drop = slim.dropout(fc2, 0.5, scope='fc2_drop')
                # 10
                logits = slim.fully_connected(fc2, 10, activation_fn=None, normalizer_fn=None, scope='d_output')
                
                softmax = tf.nn.softmax(logits, -1)
                
                return logits, softmax
    
    def build_model(self):
        # noise_z
        self.input_label = tf.placeholder(dtype=tf.float32,
                                shape=[self.batchsize, self.num_class],
                                name='input_label')
        # real data
        self.input_x = tf.placeholder(dtype=tf.float32,
                                      shape=[self.batchsize, self.input_height, self.input_width, self.input_channel],
                                      name='input_x')
        # FCN forward training phase
        self.pred_logits, self.pred_softmax = self.fully_connected_network(self.input_x, is_training=True, reuse=False)
        # FCN forward testing phase
        self.test_logits, self.test_softmax = self.fully_connected_network(self.input_x, is_training=False, reuse=True)
        
        # loss function
        def softmax_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.softmax_cross_entropy_with_logits(logits=x, targets=y)
                
        self.loss = tf.reduce_mean(softmax_cross_entropy_with_logits(self.pred_logits, self.input_label))
                        
        # optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                              beta1=self.config.beta1,
                                              beta2=self.config.beta2).minimize(self.loss)
                                              
        # visualize loss in borwser using tensorboard
        self.loss_summary = tf.summary.scalar('cross entropy loss', self.loss)
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
            # model testing per epoch
            self.test_model()

            # save model per 10 epoches
            print('epoch:{}'.format(epoch))
            if np.mod(epoch, 10) == 0:
                self.save_model()
            
            for ite in tqdm(range(50000/self.batchsize)):
                input_x, input_label = next(data_gen)
                # optimization
                _, loss, summaries = self.sess.run([self.optim, self.loss, self.summaries], feed_dict={self.input_x:input_x,
                                                                                                       self.input_label:input_label})
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
        if not self.config.is_training:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            bool_load = self.load_model()
            
            if bool_load:
                print('[***] load model successfully')
            else:
                print('[!!!] fail to load model')
                return
        
        datasource = get_data(is_training=False)
        num_batch = len(datasource.images)/self.batchsize
        
        counter = 0
        for indice in range(num_batch):
            batch_x = np.array(datasource.images[indice*self.batchsize:(indice+1)*self.batchsize])
            batch_label = np.array(datasource.labels[indice*self.batchsize:(indice+1)*self.batchsize])
            pred_softmax = self.sess.run([self.test_softmax], feed_dict={self.input_x:batch_x})
            
            counter = counter + np.sum(np.array(np.equal(np.argmax(pred_softmax[0], 1), np.argmax(batch_label, 1))).astype(np.float32))
            
        correct_rate = counter/(num_batch*self.batchsize)
        print('correct rate:{:.4f}'.format(correct_rate))
