import argparse
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

from . import read_pb, write_data
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
    parser.add_argument('-o', metavar='output directory', type=str, default='out', nargs='?',
                        help='output directory(default is out)')
    # parser.add_argument('-p', metavar='profile', type=str, nargs='?', help='specify configuration file')
    parser.add_argument('-c', action='store_true', help='output comments')

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
                model.set_config('comment', args.c)
                model.compile(args.o)
        elif path.endswith('.pb'):
            with open(path, 'rb') as fp:
                tensor = read_pb(fp)

            if args.d:
                array = numpy_helper.to_array(tensor)

                np.set_printoptions(suppress=True, threshold=sys.maxsize, linewidth=160)
                print(array)
            else:
                name = os.path.basename(path).strip('.pb') + '.data'

                with open(os.path.join(args.o, name), 'wb') as out:
                    write_data(out, tensor)

    return 0
