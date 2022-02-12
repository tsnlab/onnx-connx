from .opset import _float, _int, _string


attrset = {
    'Slice': [],
    'Resize': [_string('mode', 'nearest')],
}
