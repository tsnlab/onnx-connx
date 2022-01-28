from .opset import _int


attrset = {
    'Add': [],
    'Div': [],
    'Mul': [],
    'Relu': [],
    'Reshape': [_int('allowzero', 0)],
    'Sub': [],
}
