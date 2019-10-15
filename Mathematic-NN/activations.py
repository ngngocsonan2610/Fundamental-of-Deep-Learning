import numpy as np

# activation function and its derivative
def tanh(self,x):
    return np.tanh(x)

def tanh_prime(self,x):
    return 1-np.tanh(x)**2

def RELU(self,x):
    """for RELU activation"""
    return np.maximum(x,0)

def RELU_prime(self,x):
    """gradient through RELU function"""
    return np.sign(x)

def sigmoid(self,x):
    """sigmoid activation"""
    return 1/(1+np.exp(-x))

def sigmoid_prime(self,x):
    """gradiant through sigmoid function"""
    return x*(1-x)

def softmax(self,x):
    e_x = np.exp(x - np.max(x, axis = 0, keepdims = True))
    return e_x / e_x.sum(axis = 0)

def softmax_prime(self,x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)
    

        
        
        