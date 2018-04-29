# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.*
'''

# part1: enumerate
'''
    A new built-in function, enumerate(), will make certain loops a bit clearer. 
    enumerate(thing), where thing is either an iterator or a sequence, 
    returns a iterator that will return (0, thing[0]), (1, thing[1]), (2, thing[2]), and so forth.
    
    derived from: https://docs.python.org/2.3/whatsnew/section-enumerate.html
'''
print('part1:enumerate')
ls = range(10)

print('non-enumerate'+'-'*20)
for i in range(len(ls)):
    elem = ls[i]
    print('elem[{}]:{}'.format(i, elem))

print('enumerate'+'-'*20)
for i, elem in enumerate(ls):
    print('elem[{}]:{}'.format(i, elem))

print('\n')

# part2: zip
'''
    This function returns a list of tuples, where the i-th tuple contains
    the i-th element from each of the argument sequences or iterables. 
    The returned list is truncated in length to the length of the shortest argument sequence. 
    When there are multiple arguments which are all of the same length,
    zip() is similar to map() with an initial argument of None. 
    With a single sequence argument, it returns a list of 1-tuples. 
    With no arguments, it returns an empty list.
    
    derived from: https://docs.python.org/3.6/library/functions.html#zip
'''
print('part2: zip')
ls1 = range(0,10,2)
ls2 = range(10,20,2)

ls3 = range(5)
ls4 = range(10)

print('case1: equal length'+'-'*20)
print('ls1 length:{}'.format(len(ls1)))
print('ls2 length:{}'.format(len(ls2)))
for elem1, elem2 in zip(ls1, ls2):
    print('elem1:{}, elem2:{}'.format(elem1, elem2))

print('case2: not equal length'+'-'*20)
print('ls3 length:{}'.format(len(ls3)))
print('ls4 length:{}'.format(len(ls4)))
for elem3, elem4 in zip(ls3, ls4):
    print('elem3:{}, elem4:{}'.format(elem3, elem4))
