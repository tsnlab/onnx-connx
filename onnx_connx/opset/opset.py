import onnx


def _null(name):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.UNDEFINED
    return attr


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
    attr.strings.extends([str.encode(s, 'utf-8') for s in value])
    return attr
