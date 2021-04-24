import argparse
import onnx

from .proto import ConnxModelProto

def load_model(path) -> ConnxModelProto:
    proto = onnx.load_model(path)
    return ConnxModelProto(proto)

def compile_from_model(model_proto, path) -> int:
    connx = ConnxModelProto(model_proto)
    connx.compile(path)
    return 0

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
            pass

    return 0
