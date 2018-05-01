# coding='utf-8'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np

from tqdm import tqdm
from ops import l_relu, linear
from utils import get_data, gen_batch_data

class RNN:
    model_name = 'RNN.model'
    def __init__(self, config=None, sess=None):
        # rnn cell parameters
        self.hidden_unit_size = config.hidden_unit_size
        self.hidden_layer_size = config.hidden_layer_size
        # mnist data
        self.batchsize = config.batchsize
        self.num_class = config.num_class
        self.input_size = config.input_size
        self.time_steps = config.time_steps
        
        self.config = config
        self.sess = sess
    
    def rnn_single_cell(self, hidden_unit_size=128, scope_name='rnn_single_cell'):
        with tf.name_scope(scope_name):
            return tf.nn.rnn_cell.BasicRNNCell(hidden_unit_size)
        
    def rnn_cell(self, hidden_unit_size=128, hidden_layer_size=3, scope_name='rnn_cell'):   
        with tf.name_scope(scope_name):
            rnn_c = tf.nn.rnn_cell.MultiRNNCell([self.rnn_single_cell(hidden_unit_size) for _ in range(hidden_layer_size)])
        return rnn_c
    
    def build_model(self):
        # input labels
        self.input_label = tf.placeholder(dtype=tf.float32,
                                shape=[self.batchsize, self.num_class],
                                name='input_label')
        # input images
        self.input_x = tf.placeholder(dtype=tf.float32,
                                      shape=[self.batchsize, self.time_steps, self.input_size],
                                      name='input_x')
        
        rnn_cell = self.rnn_cell(self.hidden_unit_size, self.hidden_layer_size)
        self.initial_state = rnn_cell.zero_state(self.batchsize, tf.float32)

        # rnn forward
        '''
            self.rnn_outputs: [self.batchsize, self.time_steps, self.hidden_unit_size]
            self.final_state: [self.batchsize, self.hidden_unit_size]
        '''
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(rnn_cell, self.input_x, initial_state=self.initial_state)
        self.seqs_output = tf.reshape(self.rnn_outputs, [self.batchsize, -1])
        self.seqs_embedding = linear(self.seqs_output, self.num_class, scope='seqs_embedding')
        self.pred_seqs = tf.nn.softmax(self.seqs_embedding, 1)
        
        # loss function
        def softmax_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.softmax_cross_entropy_with_logits(logits=x, targets=y)

        self.loss = tf.reduce_mean(softmax_cross_entropy_with_logits(self.seqs_embedding, self.input_label))
        
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
            pred_seqs = self.sess.run([self.pred_seqs], feed_dict={self.input_x:batch_x})
            
            counter = counter + np.sum(np.array(np.equal(np.argmax(pred_seqs[0], 1), np.argmax(batch_label, 1))).astype(np.float32))
            
        correct_rate = counter/(num_batch*self.batchsize)
        print('correct rate:{:.4f}'.format(correct_rate))
