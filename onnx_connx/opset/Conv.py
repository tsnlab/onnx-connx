import numpy as np
from .util import Iterator
from .util import _index_to_offset

import time


def _conv(Y, y_idx, X, x_iter, W, w_iter, batch, x_channel, w_channel, feature_map, dilations):
    feature_dim = len(X.shape) - 2
    feature_shape = X.shape[2:]
    kernel_shape = W.shape[2:]

    ksize = dilations * np.array(kernel_shape)
    if dilations[0] != 1:
        ksize -= np.ones([feature_dim], dtype=np.int64)
    kernel = np.zeros(ksize, dtype=W.dtype)

    slicer = []
    for i in range(feature_dim):
        slicer.append(slice(None, None, dilations[i]))
    
    kernel[tuple(slicer)] = W[batch, w_channel]
    
    x = X[batch, x_channel]

    #print(x)
    while x_iter.next():
        x_idx = x_iter.index

        s1, ks1 = (0, -x_idx[0]) if x_idx[0] < 0 else (x_idx[0], 0)
        s2, ks2 = (0, -x_idx[1]) if x_idx[1] < 0 else (x_idx[1], 0)
        e1 = min(x.shape[0], x_idx[0] + kernel.shape[0])
        e2 = min(x.shape[1], x_idx[1] + kernel.shape[1])
        x_patch = x[s1:e1, s2:e2]

        ke1 = ks1 + x_patch.shape[0]
        ke2 = ks2 + x_patch.shape[1]
        k_patch = kernel[ks1:ke1, ks2:ke2]

        #print("Iter : ", x_idx)
        #print(x_patch)
        #print(k_patch)
        #print(f"{s1}:{e1} {s2}:{e2} / {ks1}:{ke1} {ks2}:{ke2}\n")

        y = np.sum(x_patch.flatten() * k_patch.flatten())
        
        Y[y_idx] += y
        y_idx += 1

    return y_idx


# X: (N x C x H x W) N - batch, C - channel, H, W - feature 1, 2
# W: (M x C/group x kH x kW) M is number of feature Map
# B: (M)
# Y: (M x ( C x M ) x ...
def Conv(output_count, X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides):
    start = time.time()
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
            output_shape[i] = np.floor((X.shape[2 + i] + pads[i] + pads[i + feature_dim] 
                - ((kernel_shape[i] - 1) * dilations[i] + 1)) / strides[i] + 1)

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
    
    print(time.time() - start)

    return Y
