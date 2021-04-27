import onnx

def _float(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.FLOAT
    attr.f = value
    return attr

def _int(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.INT
    attr.i = value
    return attr

def _string(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.STRING
    attr.s = str.encode(value, 'utf-8')
    return attr

def _floats(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.FLOATS
    attr.floats.extend(value)
    return attr

def _ints(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.INTS
    attr.ints.extend(value)
    return attr

def _strings(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.STRINGS
    attr.strings.extends([ str.encode(s, 'utf-8') for s in value ])
    return attr

def _index_to_offset(shape, index):
    offset = 0
    unit = 1
    for i in range(len(shape) - 1, -1, -1):
        idx = index[i]
        offset += unit * idx
        unit *= shape[i]

    return offset

__all__ = [ '_int', '_float', '_string', '_ints', '_floats', '_strings', '_index_to_offset' ]
