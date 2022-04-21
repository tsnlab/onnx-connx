import shutil
from typing import List, Tuple, Union

import connx
import numpy as np


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
            return connx.Tensor.from_nparray(input_)
        if type(input_) == str:
            # Assume it is a numpy file
            with open(input_, 'rb') as f:
                return connx.Tensor.from_numpy(np.load(f))
        else:
            raise Exception(f'Unknown input type: {type(input_)}')

    def run(self, inputs) -> Union[Tuple[np.ndarray], float]:

        inputs = [self.convert_input(input_) for input_ in inputs]
        model = connx.load_model(self.model_path)

        if self._loop_count is not None:
            return model.benchmark(inputs, self._loop_count, aggregate=True)

        outputs: List[connx.Tensor] = model.run(inputs)
        return [output.to_nparray() for output in outputs]
