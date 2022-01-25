import struct
import subprocess
import threading
import traceback
from typing import Any, Tuple

import numpy as np


WARN = '\033[93m'
END = '\033[0m'


class Stderr(threading.Thread):
    def __init__(self, stderr):
        threading.Thread.__init__(self)

        self.stderr = stderr

    def run(self):
        for line in iter(self.stderr.readline, b''):
            print(f'{WARN}stderr> ', line.decode('utf-8'), END, flush=True)


class BackendRep(object):
    def __init__(self, path):
        self.path = path

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        with subprocess.Popen(['connx', self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as proc:

            # Write number of inputs
            proc.stdin.write(struct.pack('=I', len(inputs)))

            try:
                for input in inputs:
                    # Write data
                    if type(input) == str:
                        with open(input, 'rb') as file:
                            data = file.read()
                            proc.stdin.write(data)
                    elif type(input) == np.ndarray:
                        dtype = self.get_dtype(input.dtype)
                        proc.stdin.write(struct.pack('=I', dtype))
                        proc.stdin.write(struct.pack('=I', len(input.shape)))

                        for dim in input.shape:
                            proc.stdin.write(struct.pack('=I', dim))

                        data = input.tobytes()
                        proc.stdin.write(data)
                    else:
                        raise Exception(f'Unknown input type: {type(input)}')
            except BrokenPipeError:
                print(proc.stderr.read().decode('utf-8'))
            else:
                # Terminate the connx at next loop
                proc.stdin.write(struct.pack('=i', -1))
                proc.stdin.close()

                # Print stderr first
                stderr = Stderr(proc.stderr)
                stderr.start()

                # save bytes for debugging purpose
                buf = bytearray()

                # Parse number of outputs
                try:
                    b = proc.stdout.read(4)
                    buf += b
                    count = struct.unpack('=i', b)[0]

                    if count < 0:
                        print('Error code:', count)
                        stderr.join()
                        return []

                    outputs = []

                    for i in range(count):
                        # Parse data type
                        b = proc.stdout.read(8)
                        buf += b
                        dtype, ndim = struct.unpack('=II', b)

                        shape = []
                        for i in range(ndim):
                            b = proc.stdout.read(4)
                            buf += b
                            shape.append(struct.unpack('=I', b)[0])

                        # Parse data
                        dtype = self.get_nptype(dtype)
                        itemsize = np.dtype(dtype).itemsize
                        total = self.product(shape)
                        b = proc.stdout.read(itemsize * total)
                        buf += b
                        output = np.frombuffer(b, dtype=dtype, count=self.product(shape)).reshape(shape)
                        outputs.append(output)

                    proc.stdout.close()
                except Exception as e:
                    print(f'Exception occurred: {e}')
                    traceback.print_exc()
                    print(f'Illegal input: {len(buf)} bytes')
                    print(f'as string: "{buf}"')
                    print(f'as hexa: "{buf.hex()}"')
                    return []

                stderr.join()

                return outputs

    def get_nptype(self, onnx_dtype):
        if onnx_dtype == 1:
            return np.float32
        elif onnx_dtype == 2:
            return np.uint8
        elif onnx_dtype == 3:
            return np.int8
        elif onnx_dtype == 4:
            return np.uint16
        elif onnx_dtype == 5:
            return np.int16
        elif onnx_dtype == 6:
            return np.int32
        elif onnx_dtype == 7:
            return np.int64
        elif onnx_dtype == 8:
            return str
        elif onnx_dtype == 9:
            return bool
        elif onnx_dtype == 10:
            return np.float16
        elif onnx_dtype == 11:
            return np.float64
        elif onnx_dtype == 12:
            return np.uint32
        elif onnx_dtype == 13:
            return np.uint64
        elif onnx_dtype == 14:
            return np.csingle
        elif onnx_dtype == 15:
            return np.cdouble
        else:
            raise Exception('Not supported dtype: {}'.format(onnx_dtype))

    def get_dtype(self, numpy_type):
        if numpy_type == np.float32:
            return 1
        elif numpy_type == np.uint8:
            return 2
        elif numpy_type == np.int8:
            return 3
        elif numpy_type == np.uint16:
            return 4
        elif numpy_type == np.int16:
            return 5
        elif numpy_type == np.int32:
            return 6
        elif numpy_type == np.int64:
            return 7
        elif numpy_type == str:
            return 8
        elif numpy_type == bool:
            return 9
        elif numpy_type == np.float16:
            return 10
        elif numpy_type == np.float64:
            return 11
        elif numpy_type == np.uint32:
            return 12
        elif numpy_type == np.uint64:
            return 13
        elif numpy_type == np.csingle:
            return 14
        elif numpy_type == np.cdouble:
            return 15
        else:
            raise Exception('Not supported type: {}'.format(numpy_type))

    def product(self, shape):
        p = 1
        for dim in shape:
            p *= dim

        return p
