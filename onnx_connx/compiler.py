import argparse
import os
import sys

import numpy as np
import onnx

from . import read_pb, write_data
from .proto import ConnxModelProto


def load_model(path) -> ConnxModelProto:
    proto = onnx.load_model(path)

    return ConnxModelProto(proto)


def compile_from_model(model_proto, path) -> int:
    connx = ConnxModelProto(model_proto)
    connx.compile(path)

    return 0


def print_tensor(data, depth):
    np.set_printoptions(suppress=True, threshold=sys.maxsize, linewidth=160)
    array = onnx.numpy_helper.to_array(data)
    print('\t' * depth, end='')
    print('tensor', array.dtype, array.shape)
    print('\t' * depth, end='')
    print(array)


def print_sequence(data, depth):
    print('\t' * depth, end='')
    print('sequence', data.name)

    depth += 1

    if data.elem_type == onnx.SequenceProto.DataType.TENSOR:
        for i in range(len(data.tensor_values)):
            print('\t' * depth, end='')
            print(f'[{i}] = ')
            print_tensor(data.tensor_values[i], depth + 1)
    elif data.elem_type == onnx.SequenceProto.DataType.SPARSE_TENSOR:
        for i in range(len(data.sparse_tensor_values)):
            print('\t' * depth, end='')
            print(f'[{i}] = ')
            print_tensor(data.sparse_tensor_values[i], depth + 1)
    elif data.elem_type == onnx.SequenceProto.DataType.SEQUENCE:
        for i in range(len(data.sequence_values)):
            print('\t' * depth, end='')
            print(f'[{i}] = ')
            print_sequence(data.sequence_values[i], depth + 1)
    elif data.elem_type == onnx.SequenceProto.DataType.MAP:
        for i in range(len(data.map_values)):
            print('\t' * depth, end='')
            print(f'[{i}] = ')
            print_map(data.map_values[i], depth + 1)
    elif data.elem_type == onnx.SequenceProto.DataType.OPTIONAL:
        for i in range(len(data.optional_values)):
            print('\t' * depth, end='')
            print(f'[{i}] = ')
            print_optional(data.optional_values[i], depth + 1)


def print_map(data, depth):
    print('\t' * depth, end='')
    print('map', data.name)
    print('\t' * depth, end='')
    print('keys:')
    print('\t' * depth, end='')
    print([key for key in data.keys])
    print('values:')
    print_sequence(data.values)


def print_optional(data, depth):
    print('\t' * depth, end='')
    print('optional', data.name)

    if data.elem_type == onnx.OptionalProto.DataType.TENSOR:
        if data.tensor_value is not None:
            print_tensor(data.tensor_value, depth + 1)
        else:
            print('\t' * depth, end='')
            print('None')
    elif data.elem_type == onnx.OptionalProto.DataType.SPARSE_TENSOR:
        if data.sparse_tensor_value is not None:
            print_tensor(data.sparse_tensor_value, depth + 1)
        else:
            print('\t' * depth, end='')
            print('None')
    elif data.elem_type == onnx.OptionalProto.DataType.SEQUENCE:
        if data.sequence_value is not None:
            print_sequence(data.sequence_value, depth + 1)
        else:
            print('\t' * depth, end='')
            print('None')
    elif data.elem_type == onnx.OptionalProto.DataType.MAP:
        if data.map_value is not None:
            print_map(data.map_value, depth + 1)
        else:
            print('\t' * depth, end='')
            print('None')
    elif data.elem_type == onnx.OptionalProto.DataType.OPTIONAL:
        if data.optional_value is not None:
            print_optional(data.tensor_value, depth + 1)
        else:
            print('\t' * depth, end='')
            print('None')


def run(*_args: str) -> int:
    parser = argparse.ArgumentParser(description='ONNX-CONNX Command Line Interface')
    parser.add_argument('onnx', metavar='onnx model file', nargs=1, help='an input ONNX model file or tensor pb file')
    parser.add_argument('out', metavar='output directory', nargs='?', type=str, default='out',
                        help='output directory(default is out)')
    parser.add_argument('-d', action='store_true', help='dump human readable onnx metadata to standard output')
    parser.add_argument('-c', action='store_true', help='output comments')
    # parser.add_argument('-p', metavar='profile', type=str, nargs='?', help='specify configuration file')

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
                model.compile(args.out)
        elif path.endswith('.pb'):
            with open(path, 'rb') as fp:
                data = read_pb(fp)

            if args.d:
                if isinstance(data, onnx.TensorProto):
                    print_tensor(data, 0)
                elif isinstance(data, onnx.SequenceProto):
                    print_sequence(data, 0)
                elif isinstance(data, onnx.MapProto):
                    print_map(data, 0)
                elif isinstance(data, onnx.OptionalProto):
                    print_optional(data, 0)
                else:
                    raise Exception('Unknown type: ' + type(data))
            else:
                name = os.path.basename(path).strip('.pb') + '.data'

                if isinstance(data, onnx.TensorProto):
                    with open(os.path.join(args.out, name), 'wb') as out:
                        write_data(out, data)
                elif isinstance(data, onnx.SequenceProto):
                    raise Exception('SequenceProto is not supported yet')
                elif isinstance(data, onnx.MapProto):
                    raise Exception('MapProto is not supported yet')
                elif isinstance(data, onnx.OptionalProto):
                    raise Exception('OptionalProto is not supported yet')
                else:
                    raise Exception('Unknown type: ' + type(data))

    return 0
