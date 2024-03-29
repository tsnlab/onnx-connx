from .opset import _int, _ints, _string


attrset = {
    'GreaterOrEqual': [],
    'MaxPool': [_string('auto_pad', 'NOTSET'), _int('ceil_mode', 0), _ints('dilations', []),
                _ints('kernel_shape', []), _ints('pads', []), _int('storage_order', 0), _ints('strides', [])],
    'LessOrEqual': [],
}
