import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import ModelProto, NodeProto, IR_VERSION

import tempfile

from .opset.opset import opset
from .backend_rep import BackendRep
from .compiler import compile_from_model
from .opset.opset import opset

class Backend(object):
    @classmethod
    def is_compatible(cls,
                      model,  # type: ModelProto
                      device='CPU',  # type: Text
                      **kwargs  # type: Any
                      ):  # type: (...) -> bool

        for i in range(len(model.graph.node)):
            if opset[model.graph.node[i].op_type] is None:
                return False

        return True

    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[BackendRep]
        onnx.checker.check_model(model)

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

        output = None

        if node.op_type in opset:
            op = opset[node.op_type]
            output = op(*inputs)

        return (output.dtype, output.shape, output)

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        return device in [ 'CPU', 'cpu' ]

