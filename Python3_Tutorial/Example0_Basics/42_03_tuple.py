# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.6.*
    operator
'''

import operator

# create tuple
tup = ()                            # create a none tuple
tup1 = (5,)                         # create a tuple with only an element, comma is required
tup2 = ('hello world', 1, 1994) 
tup3 = (1,2,3,4,5)
tup4 = 'a','b','c','d','e'          # default as a tuple

# access tuple
print('access tuple'+'-'*20)
single_tup2 = tup2[1]
sub_tup2 = tup2[1:]
print('single_tup2:', single_tup2)
print('sub_tup2:', sub_tup2)

print('\n')

# tuple assignment
print('tuple assignment'+'-'*20)
# tup3[0] = 1                       !!! no assignment ops
tup2_tup3 = tup2 + tup3
print('tup2_tup3:', tup2_tup3)

print('\n')

# tuple operation
print('tuple operation'+'-'*20)
tup1_plus_tup2 = tup1+tup2
tup1_multiply = tup1*5
print('tup1_plus_tup2:', tup1_plus_tup2)
print('tup1_multiply:', tup1_multiply)

print('\n')

# build-in function
print('build-in function'+'-'*20)
lt = operator.lt(tup3, tup1)        # <
print('lt:', lt)
le = operator.le(tup3, tup3)        # <=
print('le:', le)
eq = operator.eq(tup3, tup1)        # =
print('eq:', eq)
ne = operator.ne(tup3, tup3)        # !=
print('ne:', ne)
ge = operator.ge(tup3, tup1)        # >=
print('ge:', ge)
gt = operator.gt(tup3, tup3)        # >
print('gt:', gt)

print('\n')

# tuple transformation
print('tuple transformation'+'-'*20)
trans_tuple = tuple([1,1,2,2,3,3])
print('trans_tuple:', trans_tuple)

# delete tuple
del tup
del tup1
del tup2
del tup3
del tup4
