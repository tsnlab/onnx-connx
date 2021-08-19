import numpy as np


def GlobalAveragePool(output_count, X):
    spatial_shape = np.ndim(X) - 2
    Y = np.average(X, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        Y = np.expand_dims(Y, -1)
    return Y


def test():
    x = [[
           [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]],
           [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]],
          ]]

    x = np.array(x)
    print(x.shape)
    y = GlobalAveragePool(x)
    print(y)


if __name__ == "__main__":
    test()
