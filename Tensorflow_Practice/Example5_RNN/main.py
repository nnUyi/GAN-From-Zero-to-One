# coding='utf-8'
import tensorflow as tf
import os
from RNN import RNN

flags = tf.app.flags
flags.DEFINE_integer('epoch', 50, 'training epoch')
flags.DEFINE_integer('input_size', 28, 'input_size')
flags.DEFINE_integer('time_steps', 28, 'time_steps')
flags.DEFINE_integer('batchsize', 64, 'training batchsize')
flags.DEFINE_integer('num_class', 10, 'number of class')
flags.DEFINE_integer('hidden_unit_size', 128, 'hidden unit size')
flags.DEFINE_integer('hidden_layer_size', 3, 'hidden layer size')
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate')
flags.DEFINE_float('beta1', 0.5, 'beta1 of adam optimizer')
flags.DEFINE_float('beta2', 0.999, 'beta2 of adam optimizer')
flags.DEFINE_string('sample_dir', 'sample', 'sample directory')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'checkpoint directory')
flags.DEFINE_string('log_dir', 'logs', 'log directory')
flags.DEFINE_string('dataset', 'mnist', 'dataset type')
flags.DEFINE_bool('is_training', False, 'training phase')
flags.DEFINE_bool('is_testing', False, 'testing phase')

FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists(FLAGS.sample_dir):
        os.mkdir(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

def print_config():
    print('config proto:')
    print('-'*30)
    print('dataset:{}'.format(FLAGS.dataset))
    print('time steps:{}'.format(FLAGS.time_steps))
    print('input size:{}'.format(FLAGS.input_size))
    print('batchsize:{}'.format(FLAGS.batchsize))
    print('epoch:{}'.format(FLAGS.epoch))
    print('learning rate:{}'.format(FLAGS.learning_rate))
    print('beta1:{}'.format(FLAGS.beta1))
    print('beta2:{}'.format(FLAGS.beta2))
    print('training phase:{}'.format(FLAGS.is_training))
    print('testing phase:{}'.format(FLAGS.is_testing))
    print('-'*30)
    
def main(_):
    check_dir()
    print_config()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    run_option = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config = run_option) as sess:
        rnn = RNN(config=FLAGS, sess=sess)
        rnn.build_model()
        if FLAGS.is_training:
            rnn.train_model()
        if FLAGS.is_testing:
            rnn.test_model()

if __name__=='__main__':
    tf.app.run()
