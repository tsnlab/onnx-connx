from .opset import _int, _ints, _string


attrset = {
    'Conv': [_string('auto_pad', 'NOTSET'), _ints('dilations', []), _int('group', 1),
             _ints('kernel_shape', []), _ints('pads', []), _ints('strides', [])],
}
