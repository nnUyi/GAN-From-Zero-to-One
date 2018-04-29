# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/28/2018
github: https://github.com/nnUyi

Requirements:
    python3.*
    numpy
'''

import numpy as np

'''
------------------------------------------------------------------
						create array
------------------------------------------------------------------
		arange, 			(check)
		array, 				(check)
		copy, 				(check)
		empty, 				(check)
		empty_like, 		(check)
		eye, 				(check)
		fromfile, 
		fromfunction, 
		identity, 			(check)
		linspace, 			(check)
		logspace, 			(check)
		mgrid, 
		ogrid, 
		ones, 				(check)
		ones_like, 			(check)
		r , 
		zeros, 				(check)
		zeros_like 			(check)
------------------------------------------------------------------
'''
print('create array'+'-'*20)
# arange
arange_a = np.arange(1,10,2)
print('arange_a:', arange_a)

# array
a = [1,2,3,4,5]
array_a = np.array(a)
print('type a:', type(a))
print('type array_a:', type(array_a))

# copy
array_a_copy = array_a.copy()
print('array_a_copy:', array_a_copy)

# deep copy when using copy function
if array_a_copy is array_a:
	print('shallow copy')
else:
	print('deep copy')

# empty
empty_a = np.empty([3,3])						# create an empty array with size = [3,3]
print('empty_a:', empty_a)

# empty_like
empty_like_a = np.empty_like(array_a)			# create an empty array with the size of array_a
print('empty_like_a:', empty_like_a)

# eye
eye_a = np.eye(5)
print('eye_a:', eye_a)

# ones
ones_a = np.ones([5])							# create an array with size=[5] and initial value = 1
ones_b = np.ones([5,5])							# create an array with size=[5,5] and initial value = 1
print('ones_a:', ones_a)
print('ones_b:', ones_b)

# ones_like
ones_like_b = np.ones_like(ones_b)				# create an array with size equal to the size of ones_b and initial value = 1
print('ones_like_b:', ones_like_b)

# zeros
zeros_a = np.zeros([5])							# create an array with size=[5] and initial value = 0
zeros_b = np.zeros([5,5])						# create an array with size=[5,5] and initial value = 0
print('zeros_a:', zeros_a)
print('zeros_b:', zeros_b)

# zeros_like
zeros_like_b = np.zeros_like(ones_b)			# create an array with size equal to the size of zeros_b and initial value = 0
print('zeros_like_b:', zeros_like_b)

# identity
identity_a = np.identity(5, dtype=np.float32)	# create an identity array with ones on the main diagonal
print('identity_a:', identity_a)

# linspace
linspace_a = np.linspace(-1,1,20)				# create an linear array with size=[20,1] and values linear increasing between -1,1 
print('linspace_a:', linspace_a)

# logspace
logspace_a = np.logspace(-1,1,20)				# create an linear array with size=[20,1] and values linear increasing between -1,1 
print('linspace_a:', logspace_a)

# logspace equal to linspace with the following operations
linspace = np.linspace(start=-1, stop=1, num=20)
logspace = np.power(10, linspace)
print('logspace:', logspace)

print('\n')



'''
------------------------------------------------------------------
						transformation
------------------------------------------------------------------
		astype, 			(check)
		atleast 1d, 
		atleast 2d, 
		atleast 3d, 
		mat 				(check)
------------------------------------------------------------------
'''
print('transformation'+'-'*20)
# astype
b = [1,2,3,4]
array_b = np.array(b, dtype=np.int32)
print('array_b:', array_b)
astype_array_b = array_b.astype(np.float32)     # translate data type
print('astype_array_b:', astype_array_b)

# mat
c = [b,b,b,b]
print('type c:', type(c))
mat_c = np.mat(c)                               # translate data type as mat type
print('mat_c:', mat_c)

print('\n')



'''
------------------------------------------------------------------
						   consult
------------------------------------------------------------------
		all, 
		any, 
		nonzero, 
		where 
------------------------------------------------------------------
'''
print('consult'+'-'*20)
# all
all_result = np.all([[True, False],[True, True]])       # Test whether all array elements along a given axis evaluate to True.
print('all_result:', all_result)
all_result_axis0 = np.all([[True,False],[True,True]], axis=0)
print('all_result_axis0:', all_result_axis0)

# any
any_result = np.any([[True, False], [True, True]])      # Test whether any array element along a given axis evaluates to True.
print('any_result:', any_result)
any_result_axis0 = np.any([[True, False], [False, False]], axis=0)
print('any_result_axis0:', any_result_axis0)

# nonzero
d = np.array([[1,0,0], [0,2,0], [1,1,0]])
nonzero_result = np.nonzero(d)                          # return the indices of the elements that are non-zero.
print('nonzero_result:', nonzero_result)

get_nonzero_result = d[nonzero_result]
print('get_nonzero_result:', get_nonzero_result)

# where
'''
    description:
        If x and y are given and input arrays are 1-D, where is equivalent to:
        [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]
'''
where_result = np.where([[True, False], [True, True]],  # Return elements, either from x or y, depending on condition.
                                        [[1, 2], [3, 4]],
                                        [[9, 8], [7, 6]])
print('where_result:', where_result)

print('\n')



'''
------------------------------------------------------------------
							sort
------------------------------------------------------------------
		argmax, 			(check)
		argmin, 			(check)
		argsort, 			(check)
		max, 				(check)
		min, 				(check)
		ptp, 
		searchsorted, 
		sort				(check)
------------------------------------------------------------------
'''
print('sort'+'-'*20)
e = np.random.rand(10)
f = np.random.uniform(1, 10, [5,5])

# argmax
argmax_e_index = np.argmax(e)                       # returns the indices of the maximum values along an axis.
argmax_f_index = np.argmax(f, axis=0)
print('argmax_e_index:', argmax_e_index)
print('argmax_f_index:', argmax_f_index)

# argmin
argmin_e_index = np.argmin(e)                       # returns the indices of the minimum values along an axis.
argmin_f_index = np.argmin(f, axis=0)
print('argmin_e_index:', argmin_e_index)
print('argmax_f_index:', argmax_f_index)

# argsort
argsort_e_index = np.argsort(e)                     # returns the indices that would sort an array.
argsort_f_index = np.argsort(f, axis=0)
print('argsort_e_index:', argsort_e_index)
print('argsort_f_index:', argsort_f_index)

# max
max_e = np.max(e)                                   # return the maximum values along an axis
max_f = np.max(f, axis=0)
print('max_e:', max_e)
print('max_f:', max_f)

# min
min_e = np.min(e)                                   # return the minimum values along an axis
min_f = np.min(f, axis=0)
print('min_e:', min_e)
print('min_f:', min_f)

# sort
sort_e = np.sort(e)
sort_f = np.sort(f, axis=0)
print('sort_e:', sort_e)
print('sort_f:', sort_f)

print('\n')



'''	
------------------------------------------------------------------
					arithmatic operations
------------------------------------------------------------------
		choose, 
		compress, 
		cumprod, 
		cumsum, 
		inner, 
		fill, 				(check)
		imag, 
		prod, 
		put, 
		putmask, 
		real, 
		sum 				(check)
------------------------------------------------------------------
'''
print('arithmatic operations'+'-'*20)
# fill
empty_array = np.empty(2)
print('empty_array:', empty_array)
empty_array.fill(0)                                 # fill the array with a scalar value.
print('empty_array:', empty_array)

# sum
random_array_a = np.random.uniform(-1, 1, 10)       # sum the values along an axis
random_array_b = np.random.uniform(-1, 1, [5, 5])
random_array_a_sum = np.sum(random_array_a)
random_array_b_sum = np.sum(random_array_b, axis=0)
print('random_array_a_sum:', random_array_a_sum)
print('random_array_b_sum:', random_array_b_sum)

print('\n')



'''	
------------------------------------------------------------------
					  basic statistic
------------------------------------------------------------------
		convolve, 		    (check)
		cov,                (check)
		mean, 				(check)
		std, 				(check)
		var 				(check)
------------------------------------------------------------------
'''
print('basic statistic'+'-'*20)
conv_a = np.linspace(1, 10, 5)
conv_b = np.linspace(2, 11, 5)
matrix = np.random.uniform(-1, 1, [5,4])

# convolve
conv_ab = np.convolve(conv_a, conv_b)                       # returns the discrete, linear convolution of two one-dimensional sequences.
conv_ab_same = np.convolve(conv_a, conv_b, 'same')
conv_ab_valid = np.convolve(conv_a, conv_b, 'valid')
print('conv_ab:', conv_ab)
print('conv_ab_same:', conv_ab_same)
print('conv_ab_valid:', conv_ab_valid)

# mean
matrix_mean_axis0 = np.mean(matrix, axis=0)                 # returns the mean values along an axis
matrix_mean_axis1 = np.mean(matrix, axis=1)
print('matrix_mean_axis0:', matrix_mean_axis0)
print('matrix_mean_axis1:', matrix_mean_axis1)

# std
matrix_std_axis0 = np.std(matrix, axis=0)                   # returns the standard variance values along an axis 
matrix_std_axis1 = np.std(matrix, axis=1)
print('matrix_std_axis0:', matrix_std_axis0)
print('matrix_std_axis1:', matrix_std_axis1)

# var
matrix_var_axis0 = np.var(matrix, axis=0)                   # returns the variance values along an axis
matrix_var_axis1 = np.var(matrix, axis=1)
print('matrix_var_axis0:', matrix_var_axis0)
print('matrix_var_axis1:', matrix_var_axis1)

# cov
x = np.array([[0,2],[1,1],[2,0]], dtype=np.float32).T       # column first: estimate a covariance matrix, given data
cov_x = np.cov(x)
print('cov_x:', cov_x)

# how calculate covariance matrix:
mean_x = np.mean(x, axis=1)
x_minus_mean = x-np.stack([mean_x, mean_x, mean_x], axis=1)
x_cov = np.dot(x_minus_mean, x_minus_mean.T)/(x.shape[1]-1)
print('x_cov:', x_cov)

print('\n')



'''
------------------------------------------------------------------
					basic linear algebra
------------------------------------------------------------------
		cross, 
		dot, 				(check)
		outer, 
		svd, 				(check)
		vdot				(check)
------------------------------------------------------------------
'''	
print('basic linear algebra'+'-'*20)
# dot
matrix_55 = np.random.uniform(-1,1,[5,5])
matrix_54 = np.random.uniform(-1,1, [5,4])
matrix_dot = np.dot(matrix_55, matrix_54)                   # return matrix multiply between two matrixes
print('matrix_dot:', matrix_dot)

# vdot
vector_51_0 = np.array([1,2,3,4,5], dtype=np.float32)       # return the dot product of two vectors
vector_51_1 = np.array([2,3,4,5,6], dtype=np.float32)
vector_dot = np.vdot(vector_51_0, vector_51_1)
print('vector_dot:', vector_dot)

# svd
u,s,v_h = np.linalg.svd(matrix_55)                          # Singular Value Decomposition.
print('u:',u)                                               # 5*5
print('s:',s)                                               # 5*1
print('v_h:',v_h)                                           # 1*5

print('\n')
