import io
import struct
from typing import Union

import numpy as np
import onnx


def get_nptype(onnx_dtype: int):
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


def get_dtype(numpy_type: type):
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


def get_DataType(numpy_type: type):
    if numpy_type == np.float32:
        return onnx.TensorProto.DataType.FLOAT
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


def product(shape):
    p = 1
    for dim in shape:
        p *= dim

    return p


def read_pb(input_: io.RawIOBase) -> onnx.TensorProto:
    tensor = onnx.TensorProto()
    tensor.ParseFromString(input_.read())

    return tensor


def read_npy(input_: io.RawIOBase) -> np.ndarray:
    return np.load(input_)


def read_data(input_: io.RawIOBase) -> np.ndarray:
    # parse dtype
    b = input_.read(4)
    if len(b) != 4:
        raise Exception(f'Cannot read dtype. Read {len(b)} bytes only, expected 4 bytes.')
    dtype = struct.unpack('=I', b)[0]

    # parse ndim
    b = input_.read(4)
    if len(b) != 4:
        raise Exception(f'Cannot read ndim. Read {len(b)} bytes only, expected 4 bytes.')
    ndim = struct.unpack('=I', b)[0]

    # parse shape
    shape = []
    for i in range(ndim):
        b = input_.read(4)
        if len(b) != 4:
            raise Exception(f'Cannot read shape[{i}]. Read {len(b)} bytes only, expected 4 bytes.')
        shape.append(struct.unpack('=I', b)[0])

    # Parse data
    dtype = get_nptype(dtype)
    itemsize = np.dtype(dtype).itemsize
    total = product(shape)
    b = input_.read(itemsize * total)
    if len(b) != itemsize * total:
        raise Exception(f'Cannot read buffer. Read {len(b)} bytes only,'
                        f'expected {itemsize} * {total} = {itemsize * total} bytes.')
    return np.frombuffer(b, dtype=dtype, count=product(shape)).reshape(shape)


def write_npy(out: io.IOBase, array: np.array):
    np.save(out, array)


def write_data(out: io.IOBase, tensor: Union[np.ndarray, onnx.TensorProto]):
    if type(tensor) == onnx.TensorProto:
        dtype = tensor.data_type
        ndim = len(tensor.dims)
        shape = tensor.dims
        array = onnx.numpy_helper.to_array(tensor)
        buf = array.tobytes()
    elif type(tensor) == np.ndarray:
        dtype = get_dtype(tensor.dtype)
        ndim = tensor.ndim
        shape = tensor.shape
        buf = tensor.tobytes()
    else:
        raise Exception(f'Not supported tensor type: {type(tensor)}')

    size = out.write(struct.pack('=I', dtype))
    if size != 4:
        raise Exception(f'Cannot write dtype. Written {size} bytes only, expected 4 bytes.')
    size = out.write(struct.pack('=I', ndim))
    if size != 4:
        raise Exception(f'Cannot write ndim. Written {size} bytes only, expected 4 bytes.')
    for i in range(ndim):
        size = out.write(struct.pack('=I', shape[i]))
        if size != 4:
            raise Exception(f'Cannot write shape[{i}]. Written {size} bytes only, expected 4 bytes.')

    size = out.write(buf)
    if size != len(buf):
        raise Exception(f'Cannot write buffer. Written {size} bytes only, expected {len(buf)} bytes.')
