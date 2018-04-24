# coding='utf-8'
import numpy as np
import random
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data

# define datasource
class Datasource:
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels
        
# get dataset
def get_data(data_type='mnist', is_training=True):
    if data_type == 'mnist':
        raw_data = input_data.read_data_sets('./data/mnist', one_hot=True)
        shape = [28,28,1]
        if is_training:
            size = len(raw_data.train.images)
            images = np.reshape(raw_data.train.images, [size]+shape)
            labels = raw_data.train.labels
        else:
            size = len(raw_data.test.images)
            images = np.reshape(raw_data.test.images, [size]+shape)
            labels = raw_data.test.labels
    else:
        raise Exception('Error dataset')
    
    datasource = Datasource(images, labels)
    return datasource

# gen single data
def gen_data(datasource):
    while True:
        indices = range(len(datasource.images))
        random.shuffle(indices)
        for index in indices:
            image = datasource.images[index]
            label = datasource.labels[index]
            yield image, label

# gen a batch data        
def gen_batch_data(batchsize, datasource=None):
    if datasource == None:
        raise Exception('Datasource is None')
    
    data_gen = gen_data(datasource)
    while True:
        images = []
        labels = []
        for i in range(batchsize):
            image, label = next(data_gen)
            images.append(image)
            labels.append(label)
        yield np.array(images), np.array(labels)

# save image
def save_image(size, name, images):
    num_h = size[0]
    num_w = size[1]
    b,h,w,c = images.shape
    img_plane = np.zeros([num_h*h, num_w*w, c])
    for index in range(b):
        i = index % num_h
        j = index // num_w
        img_plane[i*h:(i+1)*h,j*w:(j+1)*w:] = images[index]
    scipy.misc.imsave(name, np.squeeze(img_plane))
