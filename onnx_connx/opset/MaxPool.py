import numpy as np
from .util import Iterator
from .util import _index_to_offset


import sys
# X: [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]
def MaxPool(output_count, X, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides):
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

    if auto_pad == 'NOTSET':
        if ceil_mode == 0: # ceil_mode
            for i in range(feature_dim):
                output_shape[i] = np.floor((X.shape[2 + i] + pads[i] + pads[i + feature_dim] - ((kernel_shape[i] - 1) * dilations[i] + 1)) / strides[i] + 1)
        else:
            for i in range(feature_dim):
                output_shape[i] = np.ceil((X.shape[2 + i] + pads[i] + pads[i + feature_dim] - ((kernel_shape[i] - 1) * dilations[i] + 1)) / strides[i] + 1)
    elif auto_pad == 'VALID': # auto_pad(deprecated)
        for i in range(feature_dim):
            output_shape[i] = np.ceil((X.shape[2 + i] - ((kernel_shape[i] - 1) * dilations[i] + 1) + 1) / strides[i])
    elif auto_pad == 'SAME_LOWER' or auto_pad == 'SAME_UPPER':
        for i in range(feature_dim):
            output_shape[i] = np.ceil(X.shape[2 + i] / strides[i])
            pad = (output_shape[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - X.shape[2 + i]
            pads[i] = pads[i + feature_dim] = int(pad / 2)
            if pad % 2 == 1:
                if auto_pad == 'SAME_UPPER':
                    pads[i + feature_dim] += 1
                else:
                    pads[i] += 1

    # MaxPool
    X_flatten = X.reshape(-1)
    Y = np.zeros([ X.shape[0] * X.shape[1] * int(np.prod(output_shape)) ], dtype=X.dtype)
    Indices = np.zeros([ X.shape[0] * X.shape[1] * int(np.prod(output_shape)) ], dtype=np.int64)

    y_idx = 0

    if storage_order == 1:
        if len(output_shape) >= 2:
            storage_unit = output_shape[-2] * output_shape[-1]
        else:
            storage_order = 0

    batch_unit = np.product(X.shape[1:])
    channel_unit = np.product(X.shape[2:])

    for batch in range(X.shape[0]):
        for channel in range(X.shape[1]):
            x_iter = Iterator(-pads[0:feature_dim], -pads[0:feature_dim] + output_shape * strides, strides)
            while x_iter.next():
                x_idx = x_iter.index

                y = None
                argmax_idx = None

                k_iter = Iterator([ 0 ] * len(kernel_shape), kernel_shape * dilations, dilations)
                while k_iter.next():
                    k_idx = k_iter.index
                    d_idx = x_idx + k_idx

                    if (d_idx < 0).any() or (d_idx >= feature_shape).any():
                        continue

                    # Get x at index
                    d_offset = _index_to_offset(d_idx, feature_shape)
                    x = X_flatten[batch * batch_unit + channel * channel_unit + d_offset]

                    # get maximum y
                    if y is None or x > y:
                        y = x
                        argmax_idx = d_offset#_index_to_offset(d_idx, feature_shape)

                Y[y_idx] = y

                if storage_order == 1:
                    remainder = y_idx % storage_unit
                    share = y_idx - remainder
                    a = remainder * output_shape[-1]
                    Indices[share + a % storage_unit + int(a / storage_unit)] = argmax_idx
                else:
                    Indices[y_idx] = argmax_idx

                y_idx += 1

    y_shape = X.shape[0:2] + tuple(output_shape)

    Y = Y.reshape(y_shape)
    Indices = Indices.reshape(y_shape)

    return Y, Indices
