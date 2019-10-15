import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

# cost or loss function  
def loss_softmax(Y, Yhat):    
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]
