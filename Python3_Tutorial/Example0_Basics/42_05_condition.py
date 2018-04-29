# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.6.*
'''

input_data = input('input a number:')

input_data = int(input_data)

# single condition
if input_data > 100:
    flag = True
    input_data = 100
else:
    flag = False

if flag:
    print('ACCESS flag:{}, input_data:{}'.format(flag, input_data))
else:
    print('FAIL TO ACCESS')

# multi-conditions
ops = input('input ops[0, 1, 2, 3]:')
a = 3
b = 4

if ops == 0:
    a_plus_b = a + b
    print('a_plus_b:{}'.format(a_plus_b))
elif ops == 1:
    a_minus_b = a - b
    print('a_minus_b:{}'.format(a_minus_b))
elif ops == 2:
    a_multiply_b = a*b
    print('a_multiply_b:{}'.format(a_multiply_b))
elif ops == 3:
    a_div_b = a/b
    print('a_div_b:{}'.format(a_div_b))
else:
    print('no matching ops!!!')
