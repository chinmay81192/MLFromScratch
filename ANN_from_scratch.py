
# coding: utf-8

# In[5]:


import numpy as np
from numpy import random
import matplotlib.pyplot as plt


# In[27]:


class ANN():
  def __init__(self, inp, out):
    np.random.seed(1);
    self.input = inp
    self.out = out
    self.W_ij = random.rand(3, 4)
    self.W_jk = random.rand(4, 1)
  
  def sigmoid(self,z):
    return 1/(1+np.exp(-z))

  def crossentropyerror(self,a,y):
    return -sum(y * np.log10(a) + (1-y) * np.log10(1-a))
  
  def loss_reduction(self,expected,actual):
    return -expected/actual + (1-expected)/(1-actual)
  
  def sigmoid_derivative(self,x):
    return x * (1-x)

  

  def train(self):
    loss=[100]
    i = 0
    while(loss[0]>=0.001):
        z_ij = np.dot(self.input,self.W_ij)
        a_ij = self.sigmoid(z_ij)
        z_jk = np.dot(a_ij,self.W_jk)
        a_jk = self.sigmoid(z_jk)
        loss = self.crossentropyerror(a_jk,self.out)
        #print(loss)
        dloss_jk = self.loss_reduction(self.out,a_jk)
        dsigmoid_jk = self.sigmoid_derivative(a_jk)
        dz_jk = a_ij
        #print("Gradient is {}".format(gradient_jk))
        dloss_ij = np.dot(dsigmoid_jk * dloss_jk,self.W_jk.T)
        #print("shape is {}".format(dloss_jk*dsigmoid_jk))
        dsigmoid_ij=self.sigmoid_derivative(a_ij)
        #print(dloss_ij)
        #print(dsigmoid_ij)
        dz_ij = self.input
       # print("DOT PROD IS {}".format(dsigmoid_ij*dloss_ij))
        gradient_jk = np.dot(dz_jk.T, dloss_jk*dsigmoid_jk)
        gradient_ij=np.dot(dz_ij.T,dloss_ij*dsigmoid_ij)
        self.W_ij -= 0.01 * gradient_ij
        self.W_jk -= 0.01 * gradient_jk
        i += 1
        if(i%1000 == 0):
            print ("Loss is {}".format(loss))
        #print(self.W_ij)
        #print(self.W_jk)
        
  def test(self,inp):
    z_ij = np.dot(inp,self.W_ij)
    a_ij = self.sigmoid(z_ij)
    z_jk = np.dot(a_ij,self.W_jk)
    a_jk = self.sigmoid(z_jk)
    print(a_jk)
    
    
    
    


# In[28]:


x = ANN(np.array([[0, 0, 1], 
               [0, 1, 1], 
               [1, 0, 1], 
               [0, 1, 0], 
               [1, 0, 0], 
               [1, 1, 1], 
               [0, 0, 0]]), np.array([[0, 1, 1, 1, 1, 0, 0]]).T)
x.train()


# In[39]:


x.test(np.array([[0,0,0]]))

