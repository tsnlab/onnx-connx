# -------------------------------------------------------------------------
#    onnx-connx - ONNX to CONNX Model Converter
#    Copyright (C) 2020  Semih Kim
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------

"""Backend for running ONNX on CONNX
To run this, you will need to have CONNX installed as well.
"""
import ctypes
import datetime
from typing import Optional, Any, Tuple, Sequence

import numpy
from onnx import ModelProto, NodeProto
from onnx.backend.base import Backend


class ConnxRepresentation:
    __native = ctypes.CDLL("")

    def __init__(self, *args, **kwargs):
        print('ConnxRepresentation constructor is called')
        print('  type of args->', [type(a) for a in args])
        print('  type of kwargs->', [(type(a[0]), type(a[1])) for a in kwargs.items()])
        fp = self.__native.fopen(b'/tmp/ctypes.txt', b'a')
        self.__native.fputs(str(datetime.datetime.now()).encode('utf-8'), fp)
        self.__native.fclose(fp)

    def run(self, inputs, **kwargs) -> Tuple[Any, ...]:
        print('ConnxRepresentation runner is called')
        return tuple()


class ConnxBackend(Backend):
    """
    CONNX implementation of ONNX Backend API
    """
    __SUPPORTED_MODELS = ('CPU',)

    @classmethod
    def is_compatible(cls, model: ModelProto, device: str = 'CPU', **kwargs) -> bool:
        """
        Return whether the model is compatible with the backend.
        """
        return device in ConnxBackend.__SUPPORTED_MODELS

    @classmethod
    def prepare(cls, model: ModelProto, device: str = 'CPU', **kwargs) -> ConnxRepresentation:
        """
        Load the model and creates a :class:`onnx_connx.ConnxRepresentation`
        ready to be used as a backend.

        :param model: ModelProto (returned by `onnx.load`),
            string for a filename or bytes for a serialized model
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: unused
        :return: :class:`onnx_connx.ConnxRepresentation`
        """
        if isinstance(model, ConnxRepresentation):
            return model

        return ConnxRepresentation(model, device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs: Any, device: str = 'CPU', **kwargs) -> Tuple[Any, ...]:
        """
        Compute the prediction.

        :param model: ModelProto or :class:`onnx_connx.ConnxRepresentation` returned
            by function *prepare*
        :param inputs: inputs
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: unused
        :return: predictions
        """
        return cls.prepare(model, device, **kwargs).run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node: NodeProto, inputs: Any, device: str = 'CPU',
                 outputs_info: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]] = None, **kwargs) \
            -> Optional[Tuple[Any, ...]]:
        raise NotImplementedError()

    @classmethod
    def supports_device(cls, device: str) -> bool:
        """
        Check weather specified device is supported
        """
        return device in ConnxBackend.__SUPPORTED_MODELS


is_compatible = ConnxBackend.is_compatible
prepare = ConnxBackend.prepare
run_model = ConnxBackend.run_model
run_node = ConnxBackend.run_node
supports_device = ConnxBackend.supports_device
