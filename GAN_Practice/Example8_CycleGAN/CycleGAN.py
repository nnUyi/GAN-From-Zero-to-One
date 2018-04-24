import tensorflow as tf
import numpy as np
import os
import time
from glob import glob
from ops import *
from utils import *


class CycleGAN():
    model_name='CycleGAN'
    
    def __init__(self, gf_dim=32, df_dim=32, input_channels=3,output_channels=3, input_height=256, input_width=256, is_grayscale=False, batchsize=1, lambd_l1=10, dataset_name=None, sess=None):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.input_channels = input_channels
        self.sess = sess
        
        self.batchsize = batchsize
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.is_grayscale = is_grayscale
        self.beta1 = 0.5
        self.lambd_l1 = lambd_l1
        
        self.dataset_name = dataset_name
        self.image_pool = ImagePool(maxsize=50)
        
    def generator_resnet(self, input_data_x, reuse=False, name='generator_resnet'):
        # this is an ae structure
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            print("generator_resnet")
            # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
            # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
            conv0 = tf.pad(input_data_x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
            #def conv2d(input_x, kernel_size, stride=[1,2,2,1], scope_name='conv2d', conv_type='SAME'):
            conv1 = tf.nn.relu(instance_norm(conv2d(conv0, self.gf_dim,kernel_size=7, stride=1, scope_name='g_conv1', conv_type='VALID') ,'g_conv1_in'))
            print(conv1)
            conv2 = tf.nn.relu(instance_norm(conv2d(conv1, self.gf_dim*2, kernel_size=3, stride=2, scope_name='g_conv2'), 'g_conv2_in'))
            print(conv2)
            conv3 = tf.nn.relu(instance_norm(conv2d(conv2, self.gf_dim*4, kernel_size=3, stride=2, scope_name='g_conv3'), 'g_conv3_in'))
            print(conv3)
            
            # res_block
            #def res_block(input_x, kernel_size, name='res'):
            res1 = res_block(conv3, self.gf_dim*4, name='g_res1')
            res2 = res_block(res1,  self.gf_dim*4, name='g_res2')
            res3 = res_block(res2,  self.gf_dim*4, name='g_res3')
            res4 = res_block(res3,  self.gf_dim*4, name='g_res4')
            res5 = res_block(res4,  self.gf_dim*4, name='g_res5')
            res6 = res_block(res5,  self.gf_dim*4, name='g_res6')
            res7 = res_block(res6,  self.gf_dim*4, name='g_res7')
            res8 = res_block(res7,  self.gf_dim*4, name='g_res8')
            res9 = res_block(res8,  self.gf_dim*4, name='g_res9')
            print(res9)
            
            # def deconv2d(input_x, kernel_size, output_shape, stride=[1,2,2,1], scope_name='deconv2d', deconv_type='SAME'):
            dconv1 = tf.nn.relu(instance_norm(deconv2d(res9, self.gf_dim*2, kernel_size=3, stride=2, scope_name='g_dconv1'), 'g_dconv1_in'))
            print(dconv1)
            dconv2 = tf.nn.relu(instance_norm(deconv2d(dconv1, self.gf_dim, kernel_size=3, stride=2, scope_name='g_dconv2'), 'g_dconv2_in'))
            print(dconv2)
            dconv2 = tf.pad(dconv2, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            out = tf.nn.tanh(conv2d(dconv2, self.input_channels, kernel_size=7, stride=1, scope_name='g_conv_out', conv_type='VALID'))
            print(out)
            
            return out

    def discriminator(self, input_data_x, reuse=False, name='discriminator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            print('discriminator')
            conv1 = leaky_relu(conv2d(input_data_x, self.df_dim, kernel_size=3, stride=2, scope_name='d_conv1'))
            print(conv1)
            conv2 = leaky_relu(instance_norm(conv2d(conv1, self.df_dim*2, kernel_size=3, stride=2, scope_name='d_conv2'), 'd_conv2_in'))
            print(conv2)
            conv3 = leaky_relu(instance_norm(conv2d(conv2, self.df_dim*4, kernel_size=3, stride=2, scope_name='d_conv3'), 'd_conv3_in'))
            print(conv3)
            conv4 = leaky_relu(instance_norm(conv2d(conv3, self.df_dim*8, kernel_size=3, stride=1, scope_name='d_conv4'), 'd_conv4_in'))
            print(conv4)
            conv5 = conv2d(conv4, 1, kernel_size=3, stride=1, scope_name='d_conv5')
            print(conv5)
            
            return tf.nn.sigmoid(conv5)
    
    def build_model(self):
        self.input_A = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channels], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.output_channels], name='input_B')
        
        # g loss
        self.fake_B = self.generator_resnet(self.input_A, reuse=False, name='generatorA2B')
        self.fake_A_ = self.generator_resnet(self.fake_B, reuse=False, name='generatorB2A')
        self.fake_A = self.generator_resnet(self.input_B, reuse=True, name='generatorB2A')
        self.fake_B_ = self.generator_resnet(self.fake_A, reuse=True, name='generatorA2B')
        
        self.D_fake_A = self.discriminator(self.fake_A, reuse=False, name='discriminatorA')
        self.D_fake_B = self.discriminator(self.fake_B, reuse=False, name='discriminatorB')
        
        def abs_criterion(pred, target):
            return tf.reduce_mean(tf.abs(pred-target))
        def mae_criterion(pred, target):
            return tf.reduce_mean(tf.square(pred-target))
        def sce_criterion(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        
        self.g_loss_a2b = mae_criterion(self.D_fake_B, tf.ones_like(self.D_fake_B))+ \
                            self.lambd_l1*abs_criterion(self.fake_A_, self.input_A)+ \
                            self.lambd_l1*abs_criterion(self.fake_B_, self.input_B)
        
        self.g_loss_b2a = mae_criterion(self.D_fake_A, tf.ones_like(self.D_fake_A)) + \
                            self.lambd_l1*abs_criterion(self.fake_B_, self.input_B) + \
                            self.lambd_l1*abs_criterion(self.fake_A_, self.input_A)
                            
        self.g_loss = mae_criterion(self.D_fake_B, tf.ones_like(self.D_fake_B))+ \
                            mae_criterion(self.D_fake_A, tf.ones_like(self.D_fake_A)) + \
                            self.lambd_l1*abs_criterion(self.fake_A_, self.input_A)+ \
                            self.lambd_l1*abs_criterion(self.fake_B_, self.input_B)
                            
        # d loss
        self.D_real_A = self.discriminator(self.input_A, reuse=True, name='discriminatorA')
        self.D_real_B = self.discriminator(self.input_B, reuse=True, name='discriminatorB')
        
        self.fake_A_sample = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channels], name="fake_sample_A")
        self.fake_B_sample = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channels], name="fake_sample_B")
        
        self.D_fake_sample_A = self.discriminator(self.fake_A_sample, reuse=True, name='discriminatorA')
        self.D_fake_sample_B = self.discriminator(self.fake_B_sample, reuse=True, name='discriminatorB')
        
        self.d_loss_a = (mae_criterion(self.D_real_A, tf.ones_like(self.D_real_A)) + \
                        mae_criterion(self.D_fake_sample_A, tf.zeros_like(self.D_fake_sample_A))) / 2.
        
        self.d_loss_b = (mae_criterion(self.D_real_B, tf.ones_like(self.D_real_B)) + \
                        mae_criterion(self.D_fake_sample_B, tf.zeros_like(self.D_fake_sample_B))) / 2.
        self.d_loss = self.d_loss_a + self.d_loss_b
        
        self.test_B = self.generator_resnet(self.input_A, reuse=True, name='generatorA2B')
        self.test_A = self.generator_resnet(self.input_B, reuse=True, name='generatorB2A')
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        
        for var in t_vars:
            print(var)
        
        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optimization = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
        self.g_optimization = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        # save model
        self.saver = tf.train.Saver()
        
    def train(self, config):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        counter = 1
        bool_check, counter = self.load_model(config.checkpoint_dir)
        if bool_check:
            counter = counter+1
            print("[!!!]load model successfully")
        else:
            counter = 1
            print("[***]fail to load model")
        
        start_time = time.time()
        for epoch in range(config.epochs):
            dataA = glob('./datasets/{}/*.*'.format(config.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(config.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
        
            batch_idxs = min(min(len(dataA), len(dataB)), config.train_size) // self.batchsize
            # how to choose a good learning rate
            learning_rate = config.learning_rate if epoch < config.epoch_step else config.learning_rate*((config.epochs-epoch)/(config.epochs-config.epoch_step))
            
            for idx in range(batch_idxs):
                batch_files_A = dataA[idx*self.batchsize:(idx+1)*self.batchsize]
                batch_files_B = dataB[idx*self.batchsize:(idx+1)*self.batchsize]
                # load data from datasets
                
                batch_A = [load_train_data(batch_file, config.load_size, config.fine_size) for batch_file in batch_files_A]
                batch_B = [load_train_data(batch_file, config.load_size, config.fine_size) for batch_file in batch_files_B]
                
                if (self.is_grayscale):
                    batch_A = np.array(batch_A).astype(np.float32)[:, :, :, None]
                    batch_B = np.array(batch_B).astype(np.float32)[:, :, :, None]
                else:
                    batch_A = np.array(batch_A).astype(np.float32)
                    batch_B = np.array(batch_B).astype(np.float32)
                # update generator
                _, fake_A, fake_B, g_loss = self.sess.run([self.g_optimization, self.fake_A, self.fake_B, self.g_loss], feed_dict={self.input_A:batch_A, self.input_B:batch_B, self.learning_rate:learning_rate})
                
                # fake image pool
                [fake_A, fake_B] = self.image_pool([fake_A, fake_B])
                
                # update discriminator
                _, d_loss = self.sess.run([self.d_optimization, self.d_loss], feed_dict={self.input_A:batch_A,
                                                                  self.input_B:batch_B,
                                                                  self.fake_A_sample:fake_A,
                                                                  self.fake_B_sample:fake_B,
                                                                  self.learning_rate:learning_rate})
                counter = counter+1
                iteration_time = time.time()
                print('epoch{}[{}/{}]:total_time:{:.4f},g_loss:{:.4f},d_loss:{:.4f}'.format(epoch, idx, batch_idxs,iteration_time-start_time, g_loss, d_loss))
                if np.mod(idx, 300) == 0:
                    # sample images and save them
                    self.sample_data(config, epoch, idx)
                # save model
                if np.mod(counter, 500) == 0:
                    self.save_model(config.checkpoint_dir, counter)
                
                    
    def sample_data(self, config, epoch, idx):
        sample_dataA = glob('./datasets/{}/*.*'.format(config.dataset_dir + '/testA'))
        sample_dataB = glob('./datasets/{}/*.*'.format(config.dataset_dir + '/testB'))
        np.random.shuffle(sample_dataA)
        np.random.shuffle(sample_dataB)
        sample_batch_files_A = sample_dataA[:self.batchsize]
        sample_batch_files_B = sample_dataB[:self.batchsize]
        sample_batch_A = [load_train_data(batch_file, config.load_size, config.fine_size) for batch_file in sample_batch_files_A]
        sample_batch_B = [load_train_data(batch_file, config.load_size, config.fine_size) for batch_file in sample_batch_files_B]
        
        if (self.is_grayscale):
           sample_batch_A = np.array(sample_batch_A).astype(np.float32)[:, :, :, None]
           sample_batch_B = np.array(sample_batch_B).astype(np.float32)[:, :, :, None]
        else:
           sample_batch_A = np.array(sample_batch_A).astype(np.float32)
           sample_batch_B = np.array(sample_batch_B).astype(np.float32)
        sample_A = self.sess.run(self.fake_A, feed_dict={self.input_B:sample_batch_B})
        sample_B = self.sess.run(self.fake_B, feed_dict={self.input_A:sample_batch_A})
        save_images(sample_A, [1, 1], './{}/A_train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
        save_images(sample_B, [1, 1], './{}/B_train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
    
    def test(self, config):
        bool_check, _ = self.load_model(config.checkpoint_dir)
        if bool_check:
            print("[!!!]load model successfully")
        else:
            print("[***]fail to load model")
        sample_dataA = glob('./datasets/{}/*.*'.format(config.dataset_dir + '/testA'))
        sample_dataB = glob('./datasets/{}/*.*'.format(config.dataset_dir + '/testB'))
        np.random.shuffle(sample_dataA)
        np.random.shuffle(sample_dataB)

        batch_idxs = min(min(len(sample_dataA),len(sample_dataB)), config.train_size)//self.batchsize
        for idx in range(batch_idxs):
            print('test idx:{}'.format(idx))
            sample_batch_files_A = sample_dataA[idx*self.batchsize:(idx+1)*self.batchsize]
            sample_batch_files_B = sample_dataB[idx*self.batchsize:(idx+1)*self.batchsize]
            sample_batch_A = [load_train_data(batch_file, config.load_size, config.fine_size) for batch_file in sample_batch_files_A]
            sample_batch_B = [load_train_data(batch_file, config.load_size, config.fine_size) for batch_file in sample_batch_files_B]
            
            if (self.is_grayscale):
               sample_batch_A = np.array(sample_batch_A).astype(np.float32)[:, :, :, None]
               sample_batch_B = np.array(sample_batch_B).astype(np.float32)[:, :, :, None]
            else:
               sample_batch_A = np.array(sample_batch_A).astype(np.float32)
               sample_batch_B = np.array(sample_batch_B).astype(np.float32)
            sample_A = self.sess.run(self.fake_A, feed_dict={self.input_B:sample_batch_B})
            sample_B = self.sess.run(self.fake_B, feed_dict={self.input_A:sample_batch_A})
            save_images(sample_A, [1,1], './{}/B_gen_{:04d}.png'.format(config.test_dir, idx))
            save_images(sample_batch_B, [1,1], './{}/B_test_{:04d}.png'.format(config.test_dir, idx))
            save_images(sample_B, [1,1], './{}/A_gen_{:04d}.png'.format(config.test_dir, idx))
            save_images(sample_batch_A, [1,1], './{}/A_test_{:04d}.png'.format(config.test_dir, idx))
    
    # save model            
    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batchsize)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_model(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

if __name__=="__main__":
    a = tf.Variable(tf.random_normal([64,256,256,3]))
    cyclegan = CycleGAN()
    out = cyclegan.generator_resnet(a)
    out = cyclegan.discriminator(a)
