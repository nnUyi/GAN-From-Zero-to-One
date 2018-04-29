# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/26/2018
github: https://github.com/nnUyi

Requirements:
    python3.6.*
'''

# python3
# input data from terminal
input_ = input('input data:')

# python2
# in_put = raw_input('input data:')

# output data to terminal
print('output data:{}'.format(input_))

# example
for i in range(5):
    in_ = input('input {}:'.format(i))
    print('ouput {}:{}'.format(i, in_))
