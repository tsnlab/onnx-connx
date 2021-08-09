import sys
import os
import argparse
import numpy as np
import onnx
from onnx import numpy_helper

from .proto import ConnxModelProto
from .proto import ConnxTensorProto

def load_model(path) -> ConnxModelProto:
    proto = onnx.load_model(path)

    return ConnxModelProto(proto)

def compile_from_model(model_proto, path) -> int:
    connx = ConnxModelProto(model_proto)
    connx.compile(path)

    return 0

def load_tensor(path) -> onnx.TensorProto:
    tensor = onnx.TensorProto()
    with open(path, 'rb') as f:
        tensor.ParseFromString(f.read())

    return tensor

def tensor_name(name, tensor):
    name = name.replace('_', '-')
    path = '{}_{}_{}'.format(name, str(tensor.data_type), str(len(tensor.dims)))
    for i in range(len(tensor.dims)):
        path += '_'
        path += str(tensor.dims[i])
    path += '.data'

    return path

def compile(*_args: str) -> int:
    parser = argparse.ArgumentParser(description='ONNX-CONNX Command Line Interface')
    parser.add_argument('onnx', metavar='onnx', nargs='+', help='an input ONNX model file or tensor pb file')
    parser.add_argument('-d', action='store_true', help='dump human readable onnx metadata to standard output')
    parser.add_argument('-o', metavar='output directory', type=str, default='out', nargs='?', help='output directory(default is out)')
    #parser.add_argument('-p', metavar='profile', type=str, nargs='?', help='specify configuration file')
    #parser.add_argument('-c', metavar='comment', type=str, nargs='?', choices=['true', 'false', 'True', 'False'],
    #                    help='output comments(true or false)')

    # parse args
    if len(_args) > 0:
        args = parser.parse_args(_args)
    else:
        args = parser.parse_args()

    for path in args.onnx:
        if path.endswith('.onnx'):
            model = load_model(path)

            if args.d:
                model.dump()
            else:
                model.compile(args.o)
        elif path.endswith('.pb'):
            tensor = load_tensor(path)

            if args.d:
                array = numpy_helper.to_array(tensor)

                np.set_printoptions(suppress=True, threshold=sys.maxsize, linewidth=160)
                print(array)
            else:
                name = os.path.basename(path).strip('.pb')
                name = tensor_name(name, tensor);

                array = numpy_helper.to_array(tensor)

                with open(os.path.join(args.o, name), 'wb') as out:
                    buf = array.tobytes()
                    out.write(buf)

    return 0
