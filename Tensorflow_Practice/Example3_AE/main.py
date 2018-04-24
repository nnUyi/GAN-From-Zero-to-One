# coding='utf-8'
import tensorflow as tf
import os
from AutoEncoder import AutoEncoder

flags = tf.app.flags
flags.DEFINE_integer('epoch', 30, 'training epoch')
flags.DEFINE_integer('input_channel', 1, 'input image channel')
flags.DEFINE_integer('input_height', 28, 'input image height')
flags.DEFINE_integer('input_width', 28, 'input image width')
flags.DEFINE_integer('batchsize', 64, 'training batchsize')
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
    print('input height:{}'.format(FLAGS.input_height))
    print('input width:{}'.format(FLAGS.input_width))
    print('input channel:{}'.format(FLAGS.input_channel))
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
        ae = AutoEncoder(config=FLAGS, sess=sess)
        ae.build_model()
        if FLAGS.is_training:
            ae.train_model()
        if FLAGS.is_testing:
            pass

if __name__=='__main__':
    tf.app.run()
