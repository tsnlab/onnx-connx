import shutil
import struct
import subprocess
from typing import Any, Tuple

import numpy as np

from . import read_data, write_data


class BackendRep(object):
    def __init__(self, connx_path, model_path, loop_count=None, delete_path=False):
        self.connx_path = connx_path
        self.model_path = model_path
        self._loop_count = loop_count
        self._delete_path = delete_path

    def __del__(self):
        if self._delete_path:
            shutil.rmtree(self.model_path)

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        args = [self.connx_path, self.model_path]

        if self._loop_count is not None:
            args.append('-p')
            args.append(str(self._loop_count))

        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
            # Write number of inputs
            proc.stdin.write(struct.pack('=I', len(inputs)))

            for input in inputs:
                # Write data
                if type(input) == str:
                    with open(input, 'rb') as file:
                        data = file.read()
                        proc.stdin.write(data)
                elif type(input) == np.ndarray:
                    write_data(proc.stdin, input)
                else:
                    raise Exception(f'Unknown input type: {type(input)}')

            proc.stdin.write(struct.pack('=i', -1))
            proc.stdin.flush()

            b = proc.stdout.read(4)
            if len(b) != 4:
                raise Exception(f'Cannot read output_count. Read {len(b)} bytes only, expected 4 bytes.')
            output_count = struct.unpack('=i', b)[0]

            if output_count < 0:
                raise Exception(f'Error code returned from connx: {output_count}')

            outputs = []

            for i in range(output_count):
                output = read_data(proc.stdout)
                outputs.append(output)

            return outputs
