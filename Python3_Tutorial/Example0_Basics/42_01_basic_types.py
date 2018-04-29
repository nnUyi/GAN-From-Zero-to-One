# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/26/2018
github: https://github.com/nnUyi

Requirements:
    python3.6.*
'''

# part1 show types
a = 2
print(type(a))

b = 2.5
print(type(b))

c = True
print(type(c))

d = 'hello world'
print(type(d))

# part2: basic operation
# int
a_plus_a = a+a
a_minus_a = a-a
a_div_a = a/a           # float result
a_div_div_a = a//a      # int result
a_multiply_a = a*a
a_mod_a = a%a
print('int operation'+'-'*20)
print(' plus:{} \n minus:{} \n div:{} \n div_div:{} \n multiply:{} \n mod:{}'.format(a_plus_a,
                                                                                     a_minus_a,
                                                                                     a_div_a,
                                                                                     a_div_div_a,
                                                                                     a_multiply_a,
                                                                                     a_mod_a))

print('\n')

# float
b_plus_b = b+b
b_minus_b = b-b
b_div_b = b/b           
b_div_div_b = b//b      
b_multiply_b = b*b
b_mod_b = b%b

a_plus_b = a+b          # mandatory type transformation
print('float operation'+'-'*20)
print(' plus:{} \n minus:{} \n div:{} \n div_div:{} \n multiply:{} \n mod:{} \n a_plus_b:{}'.format(b_plus_b,
                                                                                     b_minus_b,
                                                                                     b_div_b,
                                                                                     b_div_div_b,
                                                                                     b_multiply_b,
                                                                                     b_mod_b,
                                                                                     a_plus_b))

print('\n')

# bool
c_plus_c = c+c
c_minus_c = c-c
c_div_c = c/c
c_div_div_c = c//c
c_multiply_c = c*c
c_mod_c = c%c
# mixed operation
a_plus_c = a+c
b_plus_c = b+c
# logic operation
c_and_c = c and c
c_or_c = c or c
not_c = not c

print('bool operation'+'-'*20)
print(' a_plus_c:{} \n b_plus_c:{}'.format(a_plus_c, b_plus_c))

print('\n')

# string
d_plus_d = d+d

# d_minus_d = d-d           !!! no minus ops
# d_multiply_d = d*d        !!! no multiply ops
# d_div_d = d*d             !!! no div_div ops
# d_div_div_d = d*d         !!! no div ops

d_plus_number = d*3

# d[0] = 'l'                !!! no such ops, it is a constant
d_piece = d[1:]             # slice ops

print('string operation'+'-'*20)
print(' plus:{} \n multiply_number:{} \n d_piece:{}'.format(d_plus_d, d_plus_number, d_piece))
