#-*- coding:utf-8 -*-
import sys
import numpy as np


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
            10 : np.float16,
            1 : np.float32,
            11 : np.double,
            3 : np.int8,
            5 : np.int16,
            6 : np.int32,
            7 : np.int64,
            2 : np.uint8,
            4 : np.uint16,
            12 : np.uint32,
            13 : np.uint64,
            9 : bool,
            8 : str,
            16 : object,
    }

    to_type = cvtTable[to]
    from_type = input.dtype
    
    if to_type is str:
        ss = []
        for i in input.flatten():
            s = str(i).encode('utf-8')
            su = s.decode('utf-8')
            ss.append(su)        
        return np.array(ss).astype(np.object).reshape(input.shape)

    elif from_type is object or from_type is np.dtype(np.uint16) or to_type is object:
        little_endian = sys.byteorder == 'little'

        if object == to_type:
            np_uint16_view = input.astype(np.float32).flatten().view(dtype=np.uint16)
            np_bfp16 = np_uint16_view[1::2] if little_endian else np_uint16_view[0::2]
            return np_bfp16.reshape(input.shape).astype(np.uint16)
        else:
            np_bfp16 = input.flatten().view()
            np_fp32_zeros = np.zeros((len(np_bfp16) * 2,), dtype=np.uint16)
            if little_endian:
                np_fp32_zeros[1::2] = np_bfp16
            else:
                np_fp32_zeros[0::2] = np_bfp16

            np_fp32_from_bfloat = np_fp32_zeros.view(dtype=np.float32)
            return np_fp32_from_bfloat.reshape(input.shape)
    
    return input.astype(to_type)
