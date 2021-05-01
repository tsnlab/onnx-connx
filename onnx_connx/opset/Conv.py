import numpy as np
from .util import _index_to_offset
from .Iterator import Iterator

def _get(data, idx1, idx2, rest):
    idx = ((idx1,), (idx2,)) + tuple([ [i] for i in rest ])
    return data[idx][0]

def _conv(Y, y_idx, X, x_iter, W, w_iter, batch, x_channel, w_channel, feature_map, dilations):
    feature_shape = X.shape[2:]

    while x_iter.next():
        x_idx = x_iter.index

        y = 0
        while w_iter.next():
            w_idx = w_iter.index # absolute weight index
            d_idx = x_idx + w_idx * dilations # absolute x index

            if (d_idx < 0).any() or (d_idx >= feature_shape).any():
                continue

            x = _get(X, batch, x_channel, d_idx)
            w = _get(W, feature_map, w_channel, w_idx)

            y += x * w

        Y[y_idx] += y
        y_idx += 1

    return y_idx

# X: (N x C x H x W) N - batch, C - channel, H, W - feature 1, 2
# W: (M x C/group x kH x kW) M is number of feature Map
# B: (M)
# Y: (M x ( C x M ) x ...
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
            output_shape[i] = np.floor((X.shape[2 + i] + pads[i] + pads[i + feature_dim] - ((kernel_shape[i] - 1) * dilations[i] + 1)) / strides[i] + 1)

    # Conv
    Y = np.zeros([ X.shape[0] * W.shape[0] * int(np.prod(output_shape)) ], dtype=X.dtype)

    y_idx = 0
    y_unit = np.prod(output_shape)

    x_iter = Iterator(-pads[0:feature_dim], -pads[0:feature_dim] + output_shape * strides, strides)
    w_iter = Iterator((0,) * len(kernel_shape), kernel_shape, (1,) * len(kernel_shape))

    for batch in range(X.shape[0]):
        for g in range(group):
            feature_group = int(W.shape[0] / group)
            for feature_map in range(g * feature_group, (g + 1) * feature_group): # divide feature_maps into groups
                for channel in range(W.shape[1]): # iterate all of channels of feature_map
                    _conv(Y, y_idx, X, x_iter, W, w_iter, batch, g * W.shape[1] + channel, channel, feature_map, dilations)

                # Apply bias
                if B is not None:
                    Y[y_idx:y_idx + y_unit] += B[feature_map]

                # Next Y
                y_idx += y_unit

    y_shape = ( X.shape[0], W.shape[0] ) + tuple(output_shape)
    Y = Y.reshape(y_shape)

    return Y

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

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

    X = np.array([[[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                   [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                   [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                   [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]],

                  [[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                   [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                   [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
                   [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]]], dtype=np.float32)

    W = np.array([[[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]],

                  [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]],

                  [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]],

                  [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]],

                  [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]], dtype=np.float32)
    B = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    Y = Conv(X, W, B, 'NOTSET', [1], 1, [3], [0, 0], [1])
    print('Y', Y.shape)
    print(Y)
    
    X = np.array([[[ 0.61485744 , 2.266609 ,  -0.6338471 , -0.5252854 ,  0.37439212,
           -0.5790688 ,  0.00983918 , 0.9603793 , -0.46351913 , 0.36915794],
          [ 0.47667548 ,-0.71531826 ,-1.1694319 ,  2.6775472 ,  0.00551279,
            0.41795763 , 1.2194241 , -0.6988767 ,  1.2031024 ,  0.12045277],
          [-0.81724167 , 1.0358037 , -0.7069952 , -0.48040646 , 0.5303579,
            1.6195798 ,  0.5835029 ,  0.6966272 ,  0.13647288 ,-0.38006115],
          [-0.12181988 , 1.2606372 , -0.7348534 ,  0.39206776 ,-0.1505118,
            0.41955033 ,-0.2926471 , -1.883369 ,   0.9505952 , -0.6647703]],

         [[-0.6820142 ,  1.2609189 , -0.73058057 , 0.7190066 , -1.4959862,
           -1.0779321 , -0.0977203 , -1.875889 ,   1.1097682 ,  1.3101448],
          [-0.306289 ,  -1.2776446 ,  3.2391968 , -0.14821139 , 1.0438826,
           -1.0862432 , -1.4266258 , -0.31091002 ,-1.0758785 ,  0.44990197],
          [ 1.3861488 , -0.68311054 ,-0.06443063 ,-0.60621923 , 2.0141637,
           -0.31519136 , 0.51596546 , 1.0919797 , -0.8936791 ,  1.2678089],
          [ 1.3044437 ,  0.3372305 ,  0.70154065 ,-1.311863 ,   1.7068131,
            0.1054736 , -0.5890126 ,  1.4471053 , -0.41578132 , 0.08982217]]], dtype=np.float32)
    W = np.array([[[-0.01961216 ,-0.21915004 , 0.1707739],
          [ 0.11830088 ,-0.0852503 ,  0.02577502],
          [-0.21011245 , 0.04686272 , 0.00629106],
          [ 0.13839489 , 0.05055654 ,-0.18292108]],

         [[ 0.09803927 , 0.12640294 ,-0.27784857],
          [-0.08919872 , 0.13132021 , 0.04292575],
          [-0.28001052 ,-0.0465733 , -0.01670343],
          [-0.01112548 ,-0.2060661 ,  0.03494331]],

         [[-0.13500074 , 0.28724337 ,-0.27404594],
          [ 0.09046549 ,-0.09060466 ,-0.02800086],
          [-0.2243813 , -0.02602205 , 0.20020568],
          [ 0.03388408 , 0.2809816 ,  0.16553208]],

         [[ 0.03468421 , 0.19243145 , 0.2761224],
          [-0.1766825 ,  0.21740961 ,-0.20534489],
          [-0.19700938 , 0.05327201 , 0.17575413],
          [-0.00405836 ,-0.10879789 ,-0.12962887]],

         [[-0.10707444 , 0.2781434 ,  0.1961948],
          [-0.22423914 ,-0.02152416 ,-0.05354539],
          [ 0.0801405 ,  0.13197544 , 0.06836572],
          [ 0.00790668 ,-0.10484424 , 0.22352207]]], dtype=np.float32)
    B = np.array([-0.01867196 ,-0.12655136 , 0.18010029 ,-0.2809182 , -0.21195522], dtype=np.float32)

    Y = Conv(X, W, B, 'NOTSET', [1], 1, [3], [0, 0], [1])
    print('Y', Y.shape)
    print(Y)
