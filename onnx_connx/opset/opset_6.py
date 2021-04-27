from .util import *
from .MaxPool import MaxPool

opset = {
    'MaxPool': MaxPool,
}

attribute = {
    'MaxPool': [ _string('auto_pad', 'NOTSET'), _int('ceil_mode', 0), _ints('dilations', []), 
                 _ints('kernel_shape', []), _ints('pads', []), _int('storage_order', 0), _ints('strides', []) ],
}

