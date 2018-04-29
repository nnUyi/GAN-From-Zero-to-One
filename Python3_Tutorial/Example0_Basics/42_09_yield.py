# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.*
'''

# yield
'''
    The yield statement is only used when defining a generator function, 
    and is only used in the body of the generator function.
    Using a yield statement in a function definition is sufficient to cause that 
    definition to create a generator function instead of a normal function.

    When a generator function is called, it returns an iterator known as a generator iterator, 
    or more commonly, a generator. The body of the generator function is executed 
    by calling the generator's next() method repeatedly until it raises an exception.

    When a yield statement is executed, the state of the generator is frozen 
    and the value of expression_list is returned to next()'s caller. 
    By ``frozen'' we mean that all local state is retained, 
    including the current bindings of local variables, the instruction pointer, 
    and the internal evaluation stack: enough information is saved 
    so that the next time next() is invoked, the function can proceed exactly 
    as if the yield statement were just another external call.
    
    derived from: https://docs.python.org/2.4/ref/yield.html
'''

# define a datasource to store datasets
class Datasource:
    def __init__(self, data, label):
        self.data = data
        self.label = label

# generate single data
def gen_data(datasource):    
    len_data = len(datasource.data)
    while True:
        for index in range(len_data):
            # yield
            yield datasource.data[index], datasource.label[index]
        
# generate batch data
def gen_batch_data(batchsize, datasource):
    data_gen = gen_data(datasource)
    while True:
        data = []
        label = []
        for index in range(batchsize):
            data_, label_ = next(data_gen)
            data.append(data_)
            label.append(label_)
        # yield
        yield data, label

if __name__=='__main__':
    data = range(20)
    label = range(20)
    batchsize = 5
    datasource = Datasource(data, label)
    gen_batch_data = gen_batch_data(batchsize, datasource)
    for index in range(10):
        data, label = next(gen_batch_data)
        print('index {}:'.format(index))
        print('data:', data)
        print('label:', label)
        print('\n')
