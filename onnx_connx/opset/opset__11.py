from .opset import _float, _int, _ints, _string


attrset = {
    'Clip': [],
    'Conv': [_string('auto_pad', 'NOTSET'), _ints('dilations', []), _int('group', 1),
             _ints('kernel_shape', []), _ints('pads', []), _ints('strides', [])],
    'Resize': [_string('coordinate_transformation_mode', 'half_pixel'),
               _float('cubic_coeff_a', -0.75),
               _int('exclude_outside', 0),
               _float('extrapolation_value', 0.0),
               _string('mode', 'nearest'),
               _string('nearest_mode', 'round_prefer_floor')],
}
