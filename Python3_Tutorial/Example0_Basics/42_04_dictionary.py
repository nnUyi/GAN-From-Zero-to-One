# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.6.*
'''

# create dictionary
dic = {'name':'yyz', 'age':24, 'university':'FDU'}
dic1 = {'name':'JJ', 'age':27, 'university':'FDU'}
dic2 = {'name':'JJ', 'name':'wl', 'age':27, 'university':'FDU'}

# access dictionary
print('access dictionary'+'-'*20)
dic_name = dic['name']
dic_age = dic['age']
dic_university = dic['university']

dic2_name = dic2['name']                        # access the last one when encountering two same keys

print('dic_name:', dic_name)
print('dic_age:', dic_age)
print('dic_university:', dic_university)
print('dic2_name:', dic2_name)                   # access the last one when encountering two same keys

print('\n')

# dictionary assignment
print('dictionary assignment'+'-'*20)
dic['name'] = 'wl'
print('dic:', dic)

print('\n')

# delete element of dictionary
print('delete element of dictionary'+'-'*20)
del dic['name']
del dic['age']
del dic['university']
print('dic:', dic)
print('\n')

# build-in function
print('build-in function'+'-'*20)
dic_copy = dic.copy()                               # copy data from dic
print('dic_copy:', dic_copy)

dic.update(dic1)                                    # update dic according to dic1
print('dic_update:', dic)

dic_values = dic.values()                           # get values of dic
print('dic_values:', dic_values)

dic_keys = dic.keys()                               # get keys of dic
print('dic_keys:', dic_keys)

dic_items = dic.items()                             # get (key, value) items of dic
print('dic_items:', dic_items)

dic_get = dic.get('name')                           # get value of key=`name`
print('dic_get:', dic_get)

dic_new = dict.fromkeys(['name', 'age', 'university'])  # create a dictionary with keys=`['name', 'age', 'university']`
print('dic_new:', dic_new)

dic.setdefault('field', 'CVDL')                     # create new (key, value) item of dic
print('dic_setdefault:', dic)

dic_pop_value = dic.pop('field')                    # pop item with key=`field`
print('dic_pop_value', dic)

dic_popitem = dic.popitem()                         # random pop item of dic
print('dic_popitem:', dic)

dic.clear()                                         # clear all items of dic
print('dic_clear:',dic)

# delete dictionary
del dic
del dic1
del dic2
