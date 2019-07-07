# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:33:41 2017

ColumbiaX CSMM.101x Artificial Intelligence Project 3 Part 1

Usage: python problem1_3.py <input_filename> <output_filename>
<input_filename>: input1.csv, containing x1, x2 input and y label in csv format
<output_filename>: output weight1, weight2, bias in csv format
          
Passed 50/50. Comment out plot() to plot the output
"""
import argparse
import numpy as np
from sklearn.linear_model import perceptron
#import matplotlib.pyplot as plt

def load_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x = data[:,:2]
    y = data[:,2].astype(int)
    return x, y

#def plot(data, label, plane=False, weights=None, bias=None):
#    colormap = np.array(['r', 'b', 'r'])
#    plt.scatter(data[:,0],data[:,1], c=colormap[label], s=40)
#    #print(data[:,0])
#    if plane:
#        #print(weights, bias)
#        plt.ylim(-30.30)#ymin, ymax = plt.ylim()
#        plt.xlim(0,16)
#        a = -weights[1]/weights[0]
#        xx = np.linspace(1, 15)
#        yy = a * xx - bias/weights[0]
#        plt.plot(yy, xx, 'k-')    

def run():
    parser = argparse.ArgumentParser(description = "Project 3 Part I: Perceptron")
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    
    x, y = load_csv(args.input)
    target = open(args.output, 'w')
    weights, bias = None, None
    for i in range(1, 99999):
        net = perceptron.Perceptron(n_iter=i, verbose=0, fit_intercept=True)
        net.fit(x,y)
        if np.array_equal(weights, net.coef_[0]) and bias == net.intercept_:
            break
        weights = net.coef_[0]
        bias = net.intercept_
        output = np.concatenate([weights.astype(int), bias.astype(int)])
        output = ",".join(str(x) for x in output)
        target.write(output+'\n')
        
    target.close()
    return None    

if __name__ == '__main__':
    run()