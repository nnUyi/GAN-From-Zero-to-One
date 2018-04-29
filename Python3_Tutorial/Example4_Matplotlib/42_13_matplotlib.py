# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.*
    matplotlib
    numpy

References:
    1. link:https://matplotlib.org/users/screenshots.html
    2. link:https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651659306&idx=1&sn=073297703d1fc20cce350c8b8b89c172&chksm=bd4c3bb98a3bb2af1b48bdbe62e60ed52d505c7684eb9466dd71171562b4ce205b5caecbba30&mpshare=1&scene=1&srcid=0424oJZniCIKVPE2dO9i4B8l&pass_ticket=YrYjj2oo7onqTB%2BG%2FrAH4pZHONCjsd%2B7D6q6pLIbP17LEunLW0hPO9KYN6hOyEB6#rd
    
'''

import matplotlib.pyplot as plt
import numpy as np



'''
-----------------------------------------------------------------
                       plot scatter
-----------------------------------------------------------------
'''
def scatter_plot(x_data, y_data, x_label='', y_label='', title='', color='r', yscale_log=False):
    # generate points
    x_data_1 = np.random.randn(10)
    y_data_1 = np.random.randn(10)
    x_data_2 = np.random.randn(10)
    y_data_2 = np.random.randn(10)
    x_data_3 = np.random.randn(10)
    y_data_3 = np.random.randn(10)
    # create the plot object
    _, ax = plt.subplots()
    
    type1 = ax.scatter(x_data_1, y_data_1, s=50, color='r', alpha=0.5)
    type2 = ax.scatter(x_data_2, y_data_2, s=50, color='g', alpha=0.5)
    type3 = ax.scatter(x_data_3, y_data_3, s=50, color='b', alpha=0.5)
    
    if yscale_log:
        ax.set_yscale('log')
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend((type1, type2, type3), ('r', 'g', 'b'))
    ax.grid(True)
    
    plt.show()
    


'''
-----------------------------------------------------------------
                       plot line
-----------------------------------------------------------------
'''
def line_plot(x_data, y_data, x_label, y_label, title):
    # generate points
    x_data_1 = np.random.randn(10)
    y_data_1 = np.random.randn(10)
    x_data_2 = np.random.randn(10)
    y_data_2 = np.random.randn(10)
    x_data_3 = np.random.randn(10)
    y_data_3 = np.random.randn(10)
    
    # create the plot object
    _, ax = plt.subplots()
    
    # line plot
    line1 = ax.plot(x_data_1, y_data_1, lw=2, color='r', alpha=1, label='r')
    line2 = ax.plot(x_data_2, y_data_2, lw=2, color='g', alpha=1, label='g')
    line3 = ax.plot(x_data_3, y_data_3, lw=2, color='b', alpha=1, label='b')
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    
    plt.show()



'''
-----------------------------------------------------------------
                       plot histogram
-----------------------------------------------------------------
'''
def histogram_plot(x_data, y_data, x_label, y_label, bins, title):
    # generate data
    uniform_data = np.random.uniform(0,10, 100)
    gaussian_data = np.random.randn(100)
    
    # create plot object
    _, ax = plt.subplots()
    
    # plot histogram
    ax.hist(uniform_data, bins=bins, cumulative=False, color='r', alpha=0.5, label='uniform_data')
    ax.hist(gaussian_data, bins=bins, cumulative=False, color='g', alpha=0.5, label='gaussian_data')
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    
    plt.show()



'''
-----------------------------------------------------------------
                       plot bar
-----------------------------------------------------------------
'''
def bar_plot(x_data, y_data, x_label, y_label, title):
    #generate data
    x_data = range(10)
    y_data = range(10)
    
    # create plot object
    _, ax = plt.subplots()
    
    # bar plot
    ax.bar(x_data, y_data, width=0.5, align='center', color='b', alpha=1, label='gaussian_data')
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    
    plt.show()

if __name__=='__main__':
    x_data = range(10)
    y_data = range(10)
    x_label = 'x'
    y_label = 'y'
    
    scatter_plot(x_data, y_data, x_label, y_label, title='scatter plot')
    line_plot(x_data, y_data, x_label, y_label, title='line plot')
    histogram_plot(x_data, y_data, x_label, y_label, np.arange(1,10,0.09), title='histogram plot')
    bar_plot(x_data, y_data, x_label, y_label, title='bar plot')
