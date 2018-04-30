# GAN-From-Zero-to-One
  - Generative adversarial network considered as one of the hotest research fileds in deep learning has been well developed recently. This repo mainly focuses on newers, give a guidence, and provied some practical projects for learning.
  
  - In Python3_Tutorial, basics about how to use python and python libraries like scipy, Pillow, matplotlib, numpy etc are provided.
  
  - In Tensorflow_Practice, I give you some examples about how to use tensorflow to build your tensorflow projects, more details show below.
  
  - In GAN_Practice, some classical GAN models like vanilla GAN, DCGAN, CGAN, WGAN, WGAN-GP, Cycle-GAN, Pix2Pix etc. are provided.
  
# Requirements
  - tensorflow >= 1.3.0
  - numpy
  - matplotlib
  - scipy
  - tqdm

# Config
## Linux
### pip install
  - pip & python2.*
  
        $ sudo apt-get install pip
        # upgrade pip
        $ sudo pip install --upgrade pip
        
  - pip3 & python3.*
  
        $ sudo apt-get install pip
        # upgrade pip
        $ sudo pip install --upgrade pip
        
### numpy install
  - python2.*
      
        $ sudo apt-get install python-numpy
        
  - python3.*
  
        $ sudo apt-get install python3-numpy

### matplotlib install
  - python2.*
      
        $ sudo apt-get install python-matplotlib
        
  - python3.*
  
        $ sudo apt-get install python3-matplotlib


### scipy install
  - python2.*
      
        $ sudo apt-get install python-scipy
        
  - python3.*
  
        $ sudo apt-get install python3-scipy
        
### tqdm install
  - python2.*
      
        $ sudo pip install tqdm
        
  - python3.*
  
        $ sudo pip3 install tqdm

### tensorflow install
  - pip install 
  
        # cpu version: 
        $ pip install tensorflow
        # gpu version: 
        $ pip install tensorflow-gpu
        # upgrade:
        $ pip install -U tensorflow


  - conda install 
  
        $ anaconda search -t conda tensorflow
        $ anaconda show [tensorflow version]
        $ conda install --channel [the show list]
        
# File Structure
```text
Python3_Tutorial
|———　Example0_Basics
|       |———　42_00_input_ouput.py
|       |———　42_01_basic_types.py
|       |———　42_02_list.py
|       |———　42_03_tuple.py
|       |———　42_04_dictionary.py
|       |———　42_05_condition.py
|       |———　42_06_loops.py
|       |———　42_07_iteration.py
|       |———　42_08_function.py
|       |———　42_09_yield.py
|       |———　42_10_class.py
|———　Example1_Numpy
|       |———　42_11_numpy.py
|———　Example2_PIL
|       |———　42_12_PIL.py
|———　Example3_Scipy
|       |———　42_12_scipy.py
|———　Example4_Matplotlib
|       |———　42_13_matplotlib.py
------------------------------------------------
Tensorflow_Practice
|———　Example0_Basics
|     　|———　400_constant.py
|     　|———　401_variable.py	   
|     　|———　402_get_variable.py  
|　     |———　403_placeholder.py 
|　     |———　404_session.py
|　     |———　405_dataloader.py  
|  　   |———　406_optimizer.py  
|    　 |———　407_tensorboard.py  
|     　|———　408_saver.py  
|     　|———　409_simple_regression_model.py  
|———　Example1_FCN
|       |———　FCN.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example2_CNN
|       |———　CNN.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example3_AE
|       |———　AutoEncoder.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example4_GAN
|       |———　GAN.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example5_RNN
|       |———　waiting for updating
|———　Example6_DQN
|       |———　waiting for updating
------------------------------------------------
GAN_Practice
|———　Example1_GAN
|     　|———　GAN.py
|     　|———　main_GAN.py
|     　|———　ops.py  
|　     |———　utils.py
|———　Example2_CGAN
|     　|———　CGAN.py
|     　|———　main_CGAN.py
|     　|———　ops.py  
|　     |———　utils.py
|———　Example3_DCGAN
|     　|———　DCGAN.py
|     　|———　main_DCGAN.py
|     　|———　ops.py  
|　     |———　utils.py
|———　Example4_WGAN
|     　|———　WGAN.py
|     　|———　main_WGAN.py
|     　|———　ops.py  
|　     |———　utils.py
|———　Example5_WGANGP
|     　|———　WGAN_GP.py
|     　|———　main_WGAN_GP.py
|     　|———　ops.py  
|　     |———　utils.py
|———　Example6_VAEGAN
|     　|———　VAE_GAN.py
|     　|———　main_VAE_GAN.py
|     　|———　ops.py  
|　     |———　utils.py
|———　Example7_pix2pix
|     　|———　pix2pix.py
|     　|———　main.py
|     　|———　ops.py  
|　     |———　utils.py
|　     |———　dataset_download.sh             # download datasets
|———　Example8_CycleGAN
|     　|———　CycleGAN.py
|     　|———　main.py
|     　|———　ops.py  
|　     |———　utils.py
```

# Usages
## Download Repo
      
      # clone repo to local
      $ git clone https://github.com/nnUyi/Tensorflow_Practice.git
      # enter root directory
      $ cd Tensorflow_Practice
      
## Example0_Basics
      
      # In Example0_Basics, each file is individual so that you can run each .py as following
      $ python [filename.py]
      
## Example1_FCN

      # In Example1_FCN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1
      
## Example2_CNN

      # In Example2_CNN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1

## Example3_AE

      # In Example3_AE, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1

## Example4_GAN

      # In Example4_GAN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1


## Example5_RNN

  - waiting for updating
      
## Example6_DQN
  
  - waiting for updating
  
# Contact
  Email: computerscienceyyz@163.com
