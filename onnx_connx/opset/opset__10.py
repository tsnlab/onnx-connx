from .opset import _float, _int, _string


attrset = {
    'Slice': [],
    'Resize': [_string('coordinate_transformation_mode', 'half_pixel'),
               _float('cubic_coeff_a', -0.75),
               _int('exclude_outside', 0),
               _float('extrapolation_value', 0.0),
               _string('mode', 'nearest'),
               _string('nearest_mode', 'round_prefer_floor')],
}
