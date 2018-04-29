# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/26/2018
github: https://github.com/nnUyi

Requirements:
    python3.6.*
'''

l = [1, 2, 3, 4, 5]
l1 = [1, 'hello', 1.5]
l2 = [1, 2, 3, 4, 5]
print(type(l))
print('list l:', l)
print('list l1:', l1)
print('list l2:', l2)

print('\n')

# slice: left close while right open in index: [)
l_piece = l[1:2]
print('l_piece:', l_piece)

print('\n')

# assignment
l[0] = 2
l[1:] = [2,2,2,2]
print('new list:', l)

print('\n')

# lengthen
l_multiply_2 = l*3
l_plus_list = l+[3,4,5]
l_plus_l = l+l
print(' multiply :',l_multiply_2)
print(' plus_list:',l_plus_list)
print(' l_plus_l :', l_plus_l)

print('\n')

# list operation
print('list operation'+'-'*20)
l1.append('append')             # append 'meta' in the tail of list
print('append:',l1)

index = l1.index('append')      # get the index the 'append' obj
print('index:', index)

l1.insert(0, 'insert')          # insert an element before `index=0`
print('insert:', l1)

l1.pop(0)                       # pop an element in `index=0`
print('pop:', l1)

l1.remove('append')             # remove element `append`
print('remove:', l1)

l1.reverse()                    # reverse the list
print('reverse:', l1)

l1.extend([1,2,3])              # extend list 
print('extend:', l1)

count = l1.count(1)             # count the numbers of the `element 1`
print('count:', count)

l2.sort()                       # sort list with ascending order
print('sort:', l2)

# delete list
del l
del l1
del l2
