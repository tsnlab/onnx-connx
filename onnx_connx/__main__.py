import sys
import argparse
import onnx

from .proto import ConnxModelProto

def load_model(path) -> ConnxModelProto:
    proto = onnx.load_model(path)
    return ConnxModelProto(proto)

def main(*_args: str) -> object:
    parser = argparse.ArgumentParser(description='ONNX-CONNX Command Line Interface')
    parser.add_argument('onnx', metavar='onnx', nargs='+', help='an input ONNX model file or tensor pb file')
    parser.add_argument('-d', action='store_true', help='dump metadata to stdout')
    parser.add_argument('-p', metavar='profile', type=str, nargs='?', help='specify configuration file')
    parser.add_argument('-o', metavar='output', type=str, nargs='?', help='output directory(default is out)')
    parser.add_argument('-c', metavar='comment', type=str, nargs='?', choices=['true', 'false', 'True', 'False'],
                        help='output comments(true or false)')

    # parse args
    if len(_args) > 0:
        args = parser.parse_args(_args)
    else:
        args = parser.parse_args()

    for path in args.onnx:
        if path.endswith('.onnx'):
            model = load_model(path)

            if args.d:
                model.dump(sys.stdout, 0)
            else:
                pass
        elif path.endswith('.pb'):
            pass

if __name__ == '__main__':
    main()

