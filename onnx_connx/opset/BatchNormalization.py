import numpy as np


def BatchNormalization(output_count, X, scale, B, input_mean, input_var, epsilon, momentum, training_mode):
    dims_x = len(X.shape)
    dim_ones = (1,) * (dims_x - 2)
    scale = scale.reshape(-1, *dim_ones)
    B = B.reshape(-1, *dim_ones)
    input_mean = input_mean.reshape(-1, *dim_ones)
    input_var = input_var.reshape(-1, *dim_ones)

    Y = (X - input_mean) / np.sqrt(input_var + epsilon) * scale + B

    return Y


def test():
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)

    y1 = BatchNormalization(x, s, bias, mean, var, 1e-05, 0.9, 1)
    print(y1)
    y2, output_mean, output_var = BatchNormalization(x, s, bias, mean, var, 1e-05, 0.9, 1)
    print(y2, output_mean, output_var)


if __name__ == "__main__":
    test()
