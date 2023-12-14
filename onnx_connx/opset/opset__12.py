from .opset import _float, _int, _ints, _string


attrset = {
    'Celu': [_float('alpha', 1.0)],
    'GreaterOrEqual': [],
    'MaxPool': [_string('auto_pad', 'NOTSET'), _int('ceil_mode', 0), _ints('dilations', []),
                _ints('kernel_shape', []), _ints('pads', []), _int('storage_order', 0), _ints('strides', [])],
    'LessOrEqual': [],
}
