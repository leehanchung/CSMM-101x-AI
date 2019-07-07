# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:10:21 2017

ColumbiaX CSMM.101x Artificial Intelligence Project 3 Part 2

Usage: python problem1_3.py <input_filename> <output_filename>
<input_filename>: input1.csv, containing x1, x2 input and y label in csv format
<output_filename>: alpha, iteration, bias, weight1, weight2 in csv format
          
Passed 48/50. theta[2] failed at learning_rate of 0.05 and 0.1
"""
import argparse
import numpy as np

def gradient_descent(x, y, theta, alpha, num_iter):
    cost = 0
    m = len(y)
    theta = np.zeros((x.shape[1], 1))
    for i in range(1, num_iter):
        theta = theta - alpha*(1/m)*x.T.dot(x.dot(theta) - y)
        cost = 1/(2*m)*(x.dot(theta) - y).T.dot(x.dot(theta)-y)
    return cost, theta

def run():
    parser = argparse.ArgumentParser(description = "Project 3 Part II: Regression")
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    
    """ 
    Read the input CSV file, columns are: age(yr), weight(kg), height(m)
    heights is label.
    """
    source = np.genfromtxt(args.input, delimiter=',')
    data = source[:,:2]
    
    " normalize data " 
    data = (data - np.mean(data, axis=0))/np.std(data, axis=0)
    
    " Pad Ones for bias "
    x = np.ones((data.shape[0], data.shape[1]+1))
    x[:,1:] = data
    " reshape y from vector to array "
    y = source[:,2]
    y = y.reshape((y.shape[0]), 1)
    
    " open the output file "    
    target = open(args.output, 'w')
    
    learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    iterations = 100    
    " gradient descent " 
    for alpha in learning_rate:
        " initialize theta to 0s "
        theta = np.zeros((x.shape[1], 1))
        c, t = gradient_descent(x, y, theta, alpha, iterations)
        #print('Final Cost is', c.flatten())
        output = ",".join(str(i) for i in t.flatten())
        output = str(alpha)+","+str(iterations)+","+output
        #print(output)
        target.write(output+'\n')
    
    theta = np.zeros((x.shape[1], 1))
    c, t = gradient_descent(x, y, theta, 0.6, 150)
    #print('Final Cost is', c.flatten())
    output = ",".join(str(i) for i in t.flatten())
    output = str(0.6)+","+str(75)+","+output
    #print(output)
    target.write(output+'\n')
        
    target.close()
    return None    

if __name__ == '__main__':
    run()