import shutil
import struct
import subprocess
from typing import Any, List, Tuple

import numpy as np

import connx

from . import read_data, write_data


class BackendRep(object):
    def __init__(self, model_path, loop_count=None, delete_path=False):
        self.model_path = model_path
        self._loop_count = loop_count
        self._delete_path = delete_path

    def __del__(self):
        if self._delete_path:
            shutil.rmtree(self.model_path)

    def convert_input(self, input_):
        if isinstance(input_, connx.Tensor):
            return input_
        elif isinstance(input_, np.ndarray):
            print(f'from ndarray {input_.dtype}')
            return connx.Tensor.from_nparray(input_)
        if type(input_) == str:
            # Assume it is a numpy file
            with open(input_, 'rb') as f:
                return connx.Tensor.from_numpy(np.load(f))
        else:
            raise Exception(f'Unknown input type: {type(input_)}')

    def run(self, inputs):  # type: (Any, **Any) -> Tuple[Any, ...]
        if self._loop_count is not None:
            # TODO: support benchmark
            raise NotImplementedError

        inputs = [self.convert_input(input_) for input_ in inputs]
        model = connx.load_model(self.model_path)
        outputs: List[connx.Tensor] = model.run(inputs)
        return [output.to_nparray() for output in outputs]
