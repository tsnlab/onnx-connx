import os
import argparse
import tempfile
import cProfile
from typing import Tuple, Any, Text, Sequence, Dict, Optional
import numpy

from onnx import numpy_helper
import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import ModelProto, NodeProto, IR_VERSION

from .backend_rep import BackendRep
from .compiler import compile_from_model
from .opset import get_opset


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

        opset = get_opset(specs)

        for i in range(len(model.graph.node)):
            if opset[model.graph.node[i].op_type] is None:
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

        if 'out' in kwargs and kwargs['out'] is not None:
            path = kwargs['out']
            os.makedirs(path, exist_ok=True)

            compile_from_model(model, path)
            return BackendRep(path)
        else:
            with tempfile.TemporaryDirectory() as path:
                compile_from_model(model, path)
                return BackendRep(path)

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
        print('##### run_node')
        raise Exception('run_node is not implemented yet')
        '''Simple run one operator and return the results.
        Args:
            outputs_info: a list of tuples, which contains the element type and
            shape of each output. First element of the tuple is the dtype, and
            the second element is the shape. More use case can be found in
            https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py
        '''
        # TODO Remove Optional from return type
        if 'opset_version' in kwargs:
            special_context = c_checker.CheckerContext()
            special_context.ir_version = IR_VERSION
            special_context.opset_imports = {'': kwargs['opset_version']}  # type: ignore
            onnx.checker.check_node(node, special_context)
        else:
            onnx.checker.check_node(node)

        specs = [{'domain': '', 'version': 15}]  # temporary code, please use special_context
        opset = get_opset(specs)

        output = None

        if node.op_type in opset:
            op = opset[node.op_type]
            output = op(*inputs)

        return (output.dtype, output.shape, output)

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

    backend = Backend.prepare(model, out=output_dir)
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
