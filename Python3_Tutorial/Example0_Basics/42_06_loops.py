# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.6.*
'''

loop_num = 10

# loop statement
# for , while
# case: Fibonacci sequence
f0 = 1
f1 = 1

print('Fibonacci sequence:')
print('for loop:'+'-'*20)
for i in range(loop_num):
    fn = f1 + f0
    f0 = f1
    f1 = fn
    print('step {}:{}'.format(i, fn))

print('\n')

f0 = 1
f1 = 1
counter = 0
print('while loop'+'-'*20)
while counter < loop_num:
    fn = f0 + f1
    f0 = f1
    f1 = fn
    print('step {}:{}'.format(counter, fn))
    counter = counter + 1

print('\n')

# loop control statement
# continue, break, pass
# odd, even
odd = []
even = []
# obtain even number
print('continue: obtain even number'+'-'*20)
for i in range(loop_num):
    if i%2 == 0:
        even.append(i)
    else:
        continue
    print('number:{}'.format(i))
print('even number list:', even)

print('\n')

# obtain odd number
print('pass: obtain odd number'+'-'*20)
for i in range(loop_num):
    if i%2 == 0:
        pass
    else:
        odd.append(i)
    print('number:{}'.format(i))
print('odd number list:', odd)

print('\n')

odd = []
even = []
counter = 0
print('break: obtain odd and even number'+'-'*20)
while True:
    if counter%2 == 0:
        even.append(counter)
    else:
        odd.append(counter)
    counter = counter + 1
    
    if counter > 20:
        break
print('odd number:', odd)
print('even number:', even)
