import tensorflow as tf
import os

from CycleGAN import *

flags = tf.app.flags
flags.DEFINE_integer('epochs', 200, 'training epochs')
flags.DEFINE_integer('epoch_step', 100, 'epoch step')
flags.DEFINE_integer('train_size', 10000, 'maximum training size')
flags.DEFINE_bool('is_training', False, 'training or not')
flags.DEFINE_bool('is_testing', False, 'testing or not')
flags.DEFINE_string('checkpoint_dir', './checkpoint', 'checkpoint directory')
flags.DEFINE_string('sample_dir','./sample', 'sample directory')
flags.DEFINE_string('dataset_dir', '', 'dataset directory')
flags.DEFINE_string('dataset_name', '', 'dataset name')
flags.DEFINE_string('test_dir', './test', 'test directory')
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate setting')
flags.DEFINE_integer('load_size', 286, 'load image size')
flags.DEFINE_integer('fine_size', 256, 'fine image size')
FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists('./sample'):
        os.mkdir('sample')
    if not os.path.exists('./checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    if not os.path.exists('./test'):
        os.mkdir('test')

def main(_):
    check_dir()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth = True
    with tf.Session(config = run_config) as sess:
        cyclegan = CycleGAN(dataset_name=FLAGS.dataset_name, sess=sess)
        cyclegan.build_model()
        if FLAGS.is_training:
            cyclegan.train(FLAGS)
        if FLAGS.is_testing:
            cyclegan.test(FLAGS)

if __name__=='__main__':
    tf.app.run()
