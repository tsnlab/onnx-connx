import argparse
import cProfile
import os
import tempfile
from typing import Any, Dict, Optional, Sequence, Text, Tuple

import numpy
import onnx.checker
from onnx import ModelProto, NodeProto, numpy_helper

from .backend_rep import BackendRep
from .compiler import compile_from_model
from .opset import get_attrset


class Backend(object):
    @classmethod
    def is_compatible(cls,
                      model,  # type: ModelProto
                      device='CPU',  # type: Text
                      **kwargs  # type: Any
                      ):  # type: (...) -> bool

        specs = []
        for i in range(len(model.opset_import)):
            opset_import = model.opset_import[i]
            specs.append({'domain': opset_import.domain, 'version': opset_import.version})

        attrset = get_attrset(specs)

        for i in range(len(model.graph.node)):
            if model.graph.node[i].op_type not in attrset or attrset[model.graph.node[i].op_type] is None:
                # print('Not supported op_type:', model.graph.node[i].op_type)
                return False

        return True

    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[BackendRep]
        onnx.checker.check_model(model)

        if 'out' in kwargs:
            path = kwargs['out']
            os.makedirs(path, exist_ok=True)

            compile_from_model(model, path)
            return BackendRep(path)
        else:
            path = tempfile.TemporaryDirectory()
            compile_from_model(model, path.name)
            return BackendRep(path.name)

    @classmethod
    def run_model(cls,
                  model,  # type: ModelProto
                  inputs,  # type: Any
                  device='CPU',  # type: Text
                  **kwargs  # type: Any
                  ):  # type: (...) -> Tuple[Any, ...]
        backend = cls.prepare(model, device, **kwargs)
        return backend.run(inputs)

    @classmethod
    def run_node(cls,
                 node,  # type: NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Dict[Text, Any]
                 ):  # type: (...) -> Optional[Tuple[Any, ...]]
        raise Exception('run_node is not implemented yet')

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        return device in ['CPU', 'cpu']


def main(args):
    onnx_path = args.onnx[0]
    input_paths = args.pb
    output_dir = args.o

    model = onnx.load_model(onnx_path)
    inputs = []

    for input_path in input_paths:
        with open(input_path, 'rb') as f:
            tensor = onnx.TensorProto()
            tensor.ParseFromString(f.read())

            inputs.append(numpy_helper.to_array(tensor))

    kwargs = {}
    if output_dir is not None:
        kwargs['out'] = output_dir

    backend = Backend.prepare(model, *kwargs)
    outputs = backend.run(inputs)

    if type(outputs) == tuple:
        for output in outputs:
            print(output)
    else:
        print(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX Reference Backend')
    parser.add_argument('onnx', metavar='onnx', nargs=1, help='an input ONNX model file')
    parser.add_argument('pb', metavar='pb', nargs='*', help='tensor pb files')
    parser.add_argument('-o', metavar='output directory', type=str, nargs='?',
                        help='connx output directory(default is temporary directory)')
    parser.add_argument('-p', action='store_true', help='performance profiling')

    args = parser.parse_args()

    if args.p:
        print('Performance profiling...')
        cProfile.run('main(args)')
    else:
        main(args)
