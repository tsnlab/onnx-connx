from .opset import _float, _int, _ints, _string


attrset = {
    'Cast': [_int('to', 0)],
    'Clip': [],
    'Concat': [_int('axis', 0)],
    'Equal': [],
    'Exp': [],
    'Gather': [_int('axis', 0)],
    'Greater': [],
    'Less': [],
    'Log': [],
    'MatMul': [],
    'NonZero': [],
    'Resize': [_string('coordinate_transformation_mode', 'half_pixel'),
               _float('cubic_coeff_a', -0.75),
               _int('exclude_outside', 0),
               _float('extrapolation_value', 0.0),
               _string('mode', 'nearest'),
               _string('nearest_mode', 'round_prefer_floor')],
    'Sigmoid': [],
    'Sign': [],
    'Slice': [],
    'Split': [_int('axis', 0)],
    'Squeeze': [],
    'Tile': [],
    'Transpose': [_ints('perm', [])],
}
