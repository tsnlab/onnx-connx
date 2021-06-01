import numpy as np
from .util import Iterator

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
def Conv(output_count, X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides):
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
