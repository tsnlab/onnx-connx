from .opset import _float


attrset = {
    'GreaterOrEqual': [],
    'Identity': [],
    'LeakyRelu': [_float('alpha', 0.01)],
    'LessOrEqual': [],
}
