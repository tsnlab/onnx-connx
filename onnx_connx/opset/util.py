import onnx
import numpy as np


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


class Iterator:
    # start: array of start position
    # stop: array of stop position
    # step: array of step position
    # All the length must be same
    def __init__(self, start, stop, step):
        self.ndim = len(start)
        self.start = [*start]
        self.stop = [*stop]
        self.step = [*step]

        assert self.ndim == len(self.stop)
        assert self.ndim == len(self.step)

        self.index = np.array([*self.start])
        self.index[-1] -= self.step[-1]

    def next(self):
        # Go next step
        for i in range(self.ndim - 1, -1, -1):
            self.index[i] += self.step[i]
            if self.index[i] < self.stop[i]:
                return True
            else:
                self.index[i] = self.start[i]

        # Return to just before start
        self.index = np.array([*self.start])
        self.index[-1] -= self.step[-1]

        return False

    # Get offset from shape
    def offset(self, shape):
        return _index_to_offset(self.index, shape)


def _index_to_offset(index, shape):
    offset = 0
    unit = 1
    for i in range(len(shape) - 1, -1, -1):
        offset += unit * index[i]
        unit *= shape[i]

    return offset


__all__ = ['_int', '_float', '_string', '_ints', '_floats', '_strings', 'Iterator', '_index_to_offset']
