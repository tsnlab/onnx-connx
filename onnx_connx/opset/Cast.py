import sys
import numpy


def Cast(output_count, input, to):
    r"""
    Constrain input, output tensor type.
    Castable
        tensor(float16), tensor(float), tensor(double),
        tensor(int8), tensor(int16), tensor(int32), tensor(int64),
        tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64),
        tensor(bool), tensor(string), tensor(bfloat16)
    """

    cvtTable = {
        10: numpy.float16,
        1: numpy.float32,
        11: numpy.double,
        3: numpy.int8,
        5: numpy.int16,
        6: numpy.int32,
        7: numpy.int64,
        2: numpy.uint8,
        4: numpy.uint16,
        12: numpy.uint32,
        13: numpy.uint64,
        9: bool,
        8: str,
        16: object,
    }

    to_type = cvtTable[to]
    from_type = input.dtype

    if to_type is str:
        ss = []
        for i in input.flatten():
            s = str(i).encode('utf-8')
            su = s.decode('utf-8')
            ss.append(su)
        return numpy.array(ss).astype(numpy.object).reshape(input.shape)

    elif from_type is object or from_type is numpy.dtype(numpy.uint16) or to_type is object:
        little_endian = sys.byteorder == 'little'

        if object == to_type:
            np_uint16_view = input.astype(numpy.float32).flatten().view(dtype=numpy.uint16)
            np_bfp16 = np_uint16_view[1::2] if little_endian else np_uint16_view[0::2]
            return np_bfp16.reshape(input.shape).astype(numpy.uint16)
        else:
            np_bfp16 = input.flatten().view()
            np_fp32_zeros = numpy.zeros((len(np_bfp16) * 2,), dtype=numpy.uint16)
            if little_endian:
                np_fp32_zeros[1::2] = np_bfp16
            else:
                np_fp32_zeros[0::2] = np_bfp16

            np_fp32_from_bfloat = np_fp32_zeros.view(dtype=numpy.float32)
            return np_fp32_from_bfloat.reshape(input.shape)

    return input.astype(to_type)
