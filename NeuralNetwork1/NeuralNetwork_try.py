# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:15:06 2019

@author: kelvi
"""

import math

import numpy as np
np.random.seed(15)

X = np.array(([2, 9], [1, 5], [3, 6]), dtype = float)
Y = np.array(([92], [86], [89]), dtype=float)

#scaling units
X = X/np.amax(X,axis=0) #normalizing using maximum value that exists in array
Y = Y/100

class Neural_Net(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
    
        #weights
        self.W1 = np.random.randn(self.inputSize,self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize,self.outputSize)


    def forward(self,x):
        #Forward propogation
        self.z = np.dot(X,self.W1)
        self.z2 = self.sigmoidf(self.z)
        self.z3 = np.dot(self.z2,self.W2)
        o1 = self.sigmoidf(self.z3)
    
        return o1
    
    def sigmoidf(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoidPrime(self,X):
        return X*(1-X)
    
    def BackPro(self,X,y,o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        
        self.W2 += self.z2.T.dot(self.o_delta)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        
        self.W1 += X.T.dot(self.z2_delta)
        
    def Train(self,x,y):
        o = self.forward(x)
        self.BackPro(x,y,o)
        
        
NN = Neural_Net()
plotlist = []
#o = NN.forward(X)
#print('Output : \n' + str(o))
#print('Target : \n' + str(y))


for i in range(10000):
    print("Input: \n" + str(X))
    print("Actual Output : \n" + str(Y))
    print("Predicted output: \n" + str(NN.forward(X)))
    print("Loss : \n" + str(np.mean(np.square(Y - NN.forward(X)))))
    plotlist.append(NN.forward(X))
    NN.Train(X,Y)
    
    
    
        
        
