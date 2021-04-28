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
                #print('x_idx', x_idx)

                y = None
                arxmax_idx = None

                k_iter = Iterator(np.maximum(x_idx, 0), np.minimum(x_idx + kernel_shape * dilations, feature_shape), dilations)
                while k_iter.next():
                    k_idx = k_iter.index
                    #print('\tk_idx', k_idx)

                    # Get x in index (below 2 lines are numpy trick)
                    idx = tuple([ [ batch ], [ channel ] ] + [ [ i ] for i in k_idx ])
                    #print('\tidx', idx)
                    x = X[idx][0]

                    # get maximum y
                    if y is None or x > y:
                        y = x
                        argmax_idx = _index_to_offset(feature_shape, k_idx)

                Y[y_idx] = y

                y_idx += 1

    ##print(output_shape)
    y_shape = X.shape[0:2] + tuple(output_shape)

    Y = Y.reshape(y_shape)

    return Y

