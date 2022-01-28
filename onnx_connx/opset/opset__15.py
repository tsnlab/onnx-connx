from .opset import _float, _int, _null

attrset = {
    'BatchNormalization': [_float('epsilon', 1e-05), _float('momentum', 0.9), _int('training_mode', 0)],
    'Shape': [_null('end'), _int('start', 0)],
}
