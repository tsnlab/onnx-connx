import numpy as np
from .util import _index_to_offset
from .Iterator import Iterator

# X: [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]
# W: (M x C/group x kH x kW) M is number of feature map
# B: (M)
def Conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides):
    # feature dimension
    feature_dim = len(X.shape) - 2
    feature_shape = X.shape[2:]
    
    # default attribute setting
    if len(dilations) == 0:
        dilations = np.ones([ feature_dim ], dtype=np.int64)
    else:
        dilations = np.array(dilations)

    kernel_shape = np.array(kernel_shape)

    if len(pads) == 0:
        pads = np.zeros([ feature_dim * 2 ], dtype=np.int64)
    else:
        pads = np.array(pads)

    if len(strides) == 0:
        strides = np.ones([ feature_dim ], dtype=np.int64)
    else:
        strides = np.array(strides)

    # output_spatial_shape
    output_shape = np.zeros([ feature_dim ], dtype=np.int64)

    if auto_pad == 'SAME_LOWER' or auto_pad == 'SAME_UPPER':
        for i in range(feature_dim):
            output_shape[i] = np.ceil(X.shape[2 + i] / strides[i])
            pad = (output_shape[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - X.shape[2 + i]
            pads[i] = pads[i + feature_dim] = int(pad / 2)
            if pad % 2 == 1:
                if auto_pad == 'SAME_UPPER':
                    pads[i + feature_dim] += 1
                else:
                    pads[i] += 1
    else:
        for i in range(feature_dim):
            output_shape[i] = (X.shape[2 + i] - kernel_shape[i] + pads[i] + pads[i + feature_dim]) / strides[i] + 1

    #print('X.shape', X.shape)
    #print('dilations', dilations)
    #print('kernel_shape', kernel_shape)
    #print('pads', pads)
    #print('strides', strides)
    #print('ceil_mode', ceil_mode)
    #print('Y.shape', output_shape)

    # Conv
    Y = np.zeros([ X.shape[0] * X.shape[1] * int(np.prod(output_shape)) ], dtype=X.dtype)
    y_idx = 0

    for batch in range(X.shape[0]):
        for channel in range(X.shape[1]):
            x_iter = Iterator(-pads[0:feature_dim], -pads[0:feature_dim] + output_shape * strides, strides)
            while x_iter.next():
                x_idx = x_iter.index

                y = 0

                k_iter = Iterator([ 0 ] * len(kernel_shape), kernel_shape, [ 1 ] * len(kernel_shape))
                while k_iter.next():
                    k_idx = k_iter.index
                    d_idx = x_idx + k_idx * dilations

                    if (d_idx < 0).any() or (d_idx >= feature_shape).any():
                        continue

                    # Get x in index (below 2 lines are numpy trick)
                    idx = tuple([ [ batch ], [ int(channel / group) ] ] + [ [ i ] for i in d_idx ])
                    try:
                        x = X[idx][0]
                    except:
                        print('##### exception')
                        print('kernel_shape', kernel_shape)
                        print('k_idx', k_idx)
                        print('d_idx', d_idx)
                        print('idx', idx)
                        print('X.shape', X.shape)

                    # get w in index (below 2 lines are numpy trick)
                    idx = tuple([ [ batch ], [ int(channel / group) ] ] + [ [ i ] for i in k_idx ])
                    try:
                        w = W[idx][0]
                    except:
                        print('##### exception')
                        print('kernel_shape', kernel_shape)
                        print('k_idx', k_idx)
                        print('d_idx', d_idx)
                        print('idx', idx)
                        print('W.shape', W.shape)

                    # convolution
                    y += x * w

                if B is not None:
                    try:
                        y += B[int(channel / group)]
                    except:
                        print('##### B exception')
                        print('channel', channel)
                        print('group', group)
                        print('kernel_shape', kernel_shape)
                        print('k_idx', k_idx)
                        print('d_idx', d_idx)
                        print('idx', idx)
                        print('B.shape', B.shape)

                Y[y_idx] = y
                y_idx += 1

    ##print(output_shape)
    y_shape = X.shape[0:2] + tuple(output_shape)

    Y = Y.reshape(y_shape)

    return Y

if __name__ == '__main__':
    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.]]]]).astype(np.float32)
    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)

    y = Conv(x, W, None, 'NOTSET', [], 1, [3, 3], [1, 1, 1, 1], [])

    y_with_padding = np.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                                 [33., 54., 63., 72., 51.],
                                 [63., 99., 108., 117., 81.],
                                 [93., 144., 153., 162., 111.],
                                 [72., 111., 117., 123., 84.]]]]).astype(np.float32)
    print('y_ref')
    print(y_with_padding)
    print(y)

    y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                [99., 108., 117.],
                                [144., 153., 162.]]]]).astype(np.float32)

    y = Conv(x, W, None, 'NOTSET', [], 1, [3, 3], [0, 0, 0, 0], [])
    print('y_ref')
    print(y_without_padding)
    print(y)

    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                [5., 6., 7., 8., 9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.],
                [20., 21., 22., 23., 24.]]]]).astype(np.float32)
    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

    y_ref = np.array([[[[12., 27., 24.],
             [63., 108., 81.],
             [72., 117., 84.]]]]).astype(np.float32)

    y = Conv(x, W, None, 'SAME_LOWER', [], 1, [3, 3], [], [2, 2])
    print('y_ref')
    print(y_ref)
    print(y)

    x = np.array([[[[0.,  1.,  2.,  3.,  4.],  # (1, 1, 7, 5) input tensor
                    [5.,  6.,  7.,  8.,  9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.],
                    [25., 26., 27., 28., 29.],
                    [30., 31., 32., 33., 34.]]]]).astype(np.float32)
    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)
    y_with_padding = np.array([[[[12., 27., 24.],  # (1, 1, 4, 3) output tensor
                                 [63., 108., 81.],
                                 [123., 198., 141.],
                                 [112., 177., 124.]]]]).astype(np.float32)

    y = Conv(x, W, None, 'NOTSET', [], 1, [3, 3], [1, 1, 1, 1], [2, 2])

    print('y_ref')
    print(y_with_padding)
    print(y)

    y = Conv(x, W, None, 'NOTSET', [], 1, [3, 3], [0, 0, 0, 0], [2, 2])
    y_without_padding = np.array([[[[54., 72.],  # (1, 1, 3, 2) output tensor
                                    [144., 162.],
                                    [234., 252.]]]]).astype(np.float32)

    print('y_ref')
    print(y_without_padding)
    print(y)

    y = Conv(x, W, None, 'NOTSET', [], 1, [3, 3], [1, 0, 1, 0], [2, 2])
    y_with_asymmetric_padding = np.array([[[[21., 33.],  # (1, 1, 4, 2) output tensor
                                            [99., 117.],
                                            [189., 207.],
                                            [171., 183.]]]]).astype(np.float32)

    print('y_ref')
    print(y_with_asymmetric_padding)
    print(y)

