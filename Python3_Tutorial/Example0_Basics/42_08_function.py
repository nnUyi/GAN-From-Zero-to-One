# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.*
'''

# function definition
'''
    syntax:
        def function_name(parameters):
            function contents
            return [expression_list]
'''

'''
-----------------------------------------------------------------
                       function definition
-----------------------------------------------------------------
'''
# definition
print('function definition')
def fibonacci_sequence(f0, f1, n):              # function_name
    fib_seq = []                                # function contents
    fib_seq.append(f0)
    fib_seq.append(f1)
    for i in range(n):
        fn = f0 + f1
        fib_seq.append(fn)
        f0 = f1
        f1 = fn
        
    return fib_seq                              # return clause
print('\n')



'''
-----------------------------------------------------------------
                       function calling
-----------------------------------------------------------------
'''
# calling
print('function calling')
f0 = 1
f1 = 1
n = 10
fib_seq = fibonacci_sequence(f0, f1, n)
print('fibonacci_sequence:', fib_seq)
print('\n')



'''
-----------------------------------------------------------------
                       function parameters
-----------------------------------------------------------------
'''
# parameters
print('function parameters')
# part1: fixed length
print('fixed length')
def print_func1(arg1, arg2, arg3):
    print('{} {} {}'.format(arg1, arg2, arg3))

def print_func2(arg1, arg2, arg3='hello world'):
    print('{} {} {}'.format(arg1, arg2, arg3))

arg1 = 'my name is'
arg2 = 'yyz'
arg3 = 'hello world'
print('print_func1'+'-'*20)
print_func1(arg1, arg2, arg3)
print_func1(arg1, arg2=arg2, arg3=arg3)
print('print_func2'+'-'*20)
print_func2(arg1, arg2)
print_func2(arg1, arg2, arg3=arg3)

print('\n')

# part2: non-fixed length
print('non-fixed length')
def print_func3(arg1, *args):
    print(arg1)
    for arg in args:
        print(arg)
print_func3(arg1, arg2, arg3)
print('\n')



'''
-----------------------------------------------------------------
                       annonymous function
-----------------------------------------------------------------
'''
# annonymous function: lambda
print('annonymous function')
add_ops = lambda arg1, arg2: arg1+arg2
arg1 = 1
arg2 = 2
result = add_ops(arg1, arg2)
print('add_ops result:{}'.format(result))
