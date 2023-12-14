from .opset import _int, _string


attrset = {
    'Mod': [_int('fmod', 0)],
    'Resize': [_string('mode', 'nearest')],
    'Slice': [],
}
