#-*- encoding: utf-8 -*-
import numpy as np


def _BatchNormalization(X, scale, B, input_mean, input_var, epsilon, momentum):
    dims_x = len(X.shape)
    dim_ones = (1,) * (dims_x - 2)
    scale = scale.reshape(-1, *dim_ones)
    B = B.reshape(-1, *dim_ones)
    input_mean = input_mean.reshape(-1, *dim_ones)
    input_var = input_var.reshape(-1, *dim_ones)
    
    Y = (X - input_mean) / np.sqrt(input_var + epsilon) * scale + B

    return Y 

def BatchNormalization(X, scale, B, input_mean, input_var, epsilon, momentum, training_mode):    
    if training_mode == 0:
        return _BatchNormalization(X, scale, B, input_mean, input_var, epsilon, momentum)
    else:
        axis = tuple(np.delete(np.arange(len(X.shape)), 1))
        current_mean = X.mean(axis=axis)
        current_var = X.var(axis=axis)
        running_mean = input_mean * momentum + current_mean * (1 - momentum)
        running_var = input_var * momentum + current_var * (1 - momentum)
        
        Y = _BatchNormalization(X, scale, B, current_mean, current_var, epsilon, momentum)
        
        return Y, running_mean, running_var


def test():
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)

    y1 = BatchNormalization(x, s, bias, mean, var, 1e-05, 0.9, 1)
    y2, output_mean, output_var = BatchNormalization(x, s, bias, mean, var, 1e-05, 0.9, 1)


if __name__ == "__main__":
    test()
