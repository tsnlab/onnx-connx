import onnx
import numpy as np

def create_attr_int(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.INT
    attr.i = value

    return attr

def create_attr_float(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.FLOAT
    attr.f = value

    return attr

def create_attr_string(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.STRING
    attr.s = value.encode()

    return attr

def create_attr_ints(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.INTS

    for v in value:
        attr.ints.append(v)

    return attr

def create_attr_floats(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.FLOATS

    for v in value:
        attr.floats.append(v)

    return attr

def create_attr_strings(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.STRINGS

    for v in value:
        attr.strings.append(v.encode())

    return attr

def create_attr_null(name):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.UNDEFINED

    return attr

_normalizers = {}

_normalizers['Add'] = ()

_normalizers['BatchNormalization'] = (
    'epsilon', 'float', 0.00001,
    'momentum', 'float', 0.9,
)

_normalizers['Cast'] = (
    'to', 'int', None,
)

_normalizers['Ceil'] = ()

_normalizers['Concat'] = (
    'axis', 'int', None
)

def _valid_pad(attrs):
    # kernel_shape
    kernel_shape = attrs[3]

    # auto_pad
    auto_pad = attrs[0]

    # pads
    pads = attrs[4]
    if auto_pad.s == 'VALID':
        if pads == None:
            attrs[4] = create_attr_floats('pads', np.zeros(len(kernel_shape.ints) * 2, np.int32))
        else:
            for i in range(len(kernel_shape.ints) * 2):
                pads.ints[i] = 0

_normalizers['Conv'] = (
    'auto_pad', 'string', 'NOTSET', 
    'dilations', 'ints', lambda attrs: np.ones(len(attrs[3].ints), np.int32),
    'group', 'int', 1,
    'kernel_shape', 'ints', None,
    'pads', 'ints', lambda attrs: np.ones(len(attrs[3].ints) * 2, np.int32),
    'strides', 'ints', lambda attrs: np.ones(len(attrs[3].ints), np.int32),
    _valid_pad
)

_normalizers['Div'] = ()

_normalizers['Exp'] = ()

_normalizers['GlobalAveragePool'] = ()

_normalizers['Identity'] = ()

_normalizers['LeakyRelu'] = (
    'alpha', 'float', 0.01
)

_normalizers['Loop'] = (
    'body', 'graph', None
)

_normalizers['MatMul'] = ()

_normalizers['MaxPool'] = (
    'auto_pad', 'string', 'NOTSET', 
    'ceil_mode', 'int', 0,
    'dilations', 'ints', lambda attrs: np.ones(len(attrs[3].ints), np.int32),
    'kernel_shape', 'ints', None,
    'pads', 'ints', lambda attrs: np.zeros(len(attrs[3].ints) * 2, np.int32),
    'storage_order', int, 0,
    'strides', 'ints', lambda attrs: np.ones(len(attrs[3].ints), np.int32),
    _valid_pad
)

_normalizers['Mul'] = ()

_normalizers['NonMaxSuppression'] = (
    'center_point_box', 'int', 0,
)

_normalizers['ReduceMin'] = (
    'axes', 'ints', None,
    'keepdims', 'int', 0,
)

_normalizers['Relu'] = ()

_normalizers['Reshape'] = ()

_normalizers['Resize'] = (
    'coordinate_transformation_mode', 'string', 'half_pixel',
    'cubic_coeff_a', 'float', -0.75,
    'exclude_outside', 'int', 0,
    'extrapolation_value', 'float', 0.0,
    'mode', 'string', 'nearest',
    'nearest_mode', 'string', 'round_prefer_floor',
)

_normalizers['Round'] = ()

_normalizers['Shape'] = ()

_normalizers['Sigmoid'] = ()

_normalizers['Slice'] = ()

_normalizers['Squeeze'] = (
    'axes', 'ints', None
)

_normalizers['Sub'] = ()

_normalizers['Tile'] = ()

_normalizers['Transpose'] = (
    'perm', 'ints', None,
)

_normalizers['Unsqueeze'] = (
    'axes', 'ints', None,
)

def normalize(op, attrs):
    def get_attr(name):
        for attr in attrs:
            if attr.name == name:
                return attr

        return None

    normalizer = _normalizers[op]
    if normalizer is None:
        return attrs

    result = []
    length = len(normalizer)

    # explicit attrs
    for i in range(0, length, 3):
        if type(normalizer[i]) is str:
            attr = get_attr(normalizer[i])
            if attr is None and not hasattr(normalizer[i + 2], '__call__'):
                if normalizer[i + 1] == 'int':
                    attr = create_attr_int(op, normalizer[i + 2])
                elif normalizer[i + 1] == 'float':
                    attr = create_attr_float(op, normalizer[i + 2])
                elif normalizer[i + 1] == 'string':
                    attr = create_attr_string(op, normalizer[i + 2])
                elif normalizer[i + 1] == 'ints':
                    attr = create_attr_ints(op, normalizer[i + 2])
                elif normalizer[i + 1] == 'floats':
                    attr = create_attr_floats(op, normalizer[i + 2])
                elif normalizer[i + 1] == 'strings':
                    attr = create_attr_strings(op, normalizer[i + 2])

            result.append(attr)
        elif i + 2 < length:
            result.append(None)

    # implicit attrs
    for i in range(0, length, 3):
        if type(normalizer[i]) is str:
            if result[int(i / 3)] is None and i + 2 < length and hasattr(normalizer[i + 2], '__call__'):
                if normalizer[i + 1] == 'int':
                    value = normalizer[i + 2](result)
                    attr = create_attr_int(op, value)
                elif normalizer[i + 1] == 'float':
                    value = normalizer[i + 2](result)
                    attr = create_attr_float(op, value)
                elif normalizer[i + 1] == 'string':
                    value = normalizer[i + 2](result)
                    attr = create_attr_string(op, value)
                elif normalizer[i + 1] == 'ints':
                    value = normalizer[i + 2](result)
                    attr = create_attr_ints(op, value)
                elif normalizer[i + 1] == 'floats':
                    value = normalizer[i + 2](result)
                    attr = create_attr_floats(op, value)
                elif normalizer[i + 1] == 'strings':
                    value = normalizer[i + 2](result)
                    attr = create_attr_strings(op, value)

                result[int(i / 3)] = attr
        else:
            normalizer[i](result)

    # null attr
    for i in range(len(result)):
        if result[i] is None:
            result[i] = create_attr_null(normalizer[i * 3])

    return result
