import sys
import os
import onnx
from .attr import default_attribute

class ConnxObject:
    def __init__(self, proto=None, parent=None):
        self.proto = proto
        self.parent = parent

    def _tab(self, out, depth):
        for i in range(depth):
            out.write('\t')

    def dump(self, depth):
        pass

    def to_connx_text(self):
        pass

    def to_connx_binary(self):
        pass


class ConnxModelProto(ConnxObject):
    def __init__(self, proto):
        super().__init__(proto)

        self.graph = ConnxGraphProto(proto.graph, self)

    def dump(self, depth=0):
        out = sys.stdout

        self._tab(out, depth)
        out.write('ModelProto\n')

        self.graph.dump(depth + 1)

    def to_connx_text(self, path):
        os.makedirs(path, exist_ok=True)

        self.graph.to_connx_text(os.path.join(path, 'main.connx'))


class ConnxGraphProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        self.initializer = [ ConnxTensorProto(proto, self) for proto in proto.initializer ]
        self.sparse_initializer = [ ConnxSparseTensorProto(proto, self) for proto in proto.sparse_initializer ]
        self.input = [ ]
        self.output = [ ]
        self.value_info = [ ConnxValueInfoProto(proto, self) for proto in proto.value_info ]

        # Make value_info reflects to initializer
        for initializer in self.initializer:
            value_info = self.get_value_info(initializer.proto.name)

            if value_info is None:
                value_info_proto = onnx.ValueInfoProto()
                value_info_proto.name = initializer.proto.name
                value_info_proto.type.tensor_type.elem_type = initializer.proto.data_type
                for i in range(len(initializer.proto.dims)):
                    dim = onnx.TensorShapeProto.Dimension()
                    dim.dim_value = initializer.proto.dims[i]
                    value_info_proto.type.tensor_type.shape.dim.append(dim)

                value_info = ConnxValueInfoProto(value_info_proto, self)
                value_info.initializer = initializer
                self.value_info.append(value_info)

        # Make value_info reflects to sparse_initializer
        for sparse_initializer in self.sparse_initializer:
            value_info = self.get_value_info(sparse_initializer.proto.name)

            if value_info is None:
                value_info_proto = onnx.ValueInfoProto()
                value_info_proto.name = sparse_initializer.proto.name
                value_info_proto.type.tensor_type.elem_type = sparse_initializer.proto.data_type
                for i in range(len(sparse_initializer.proto.dims)):
                    dim = onnx.TensorShapeProto.Dimension()
                    dim.dim_value = sparse_initializer.proto.dims[i]
                    value_info_proto.type.tensor_type.shape.dim.append(dim)

                value_info = ConnxValueInfoProto(value_info_proto, self)
                value_info.sparse_initializer = sparse_initializer
                self.value_info.append(value_info)

        # Make value_info reflects to input
        for input in proto.input:
            value_info = self.get_value_info(input.name)

            if value_info is None:
                value_info = ConnxValueInfoProto(input, self)
                self.value_info.append(value_info)

            self.input.append(value_info)

        # Make value_info reflects to output
        for output in proto.output:
            value_info = self.get_value_info(output.name)

            if value_info is None:
                value_info = ConnxValueInfoProto(output, self)
                self.value_info.append(value_info)

            self.output.append(value_info)

        self.node = [ ConnxNodeProto(proto, self) for proto in proto.node ]

    def get_value_info(self, name):
        for value_info in self.value_info:
            if value_info.proto.name == name:
                return value_info

        return None

    def get_value_info(self, name):
        for value_info in self.value_info:
            if value_info.proto.name == name:
                return value_info

        return None

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('GraphProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        out.write(self.proto.name)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('initializer\n')

        for initializer in self.initializer:
            initializer.dump(depth + 2)

        self._tab(out, depth + 1)
        out.write('sparse_initializer\n')

        for sparse_initializer in self.sparse_initializer:
            sparse_initializer.dump(depth + 2)

        self._tab(out, depth + 1)
        out.write('input\n')

        for input in self.input:
            input.dump(depth + 2)

        self._tab(out, depth + 1)
        out.write('output\n')

        for output in self.output:
            output.dump(depth + 2)

        self._tab(out, depth + 1)
        out.write('value_info\n')

        for value_info in self.value_info:
            value_info.dump(depth + 2)

        self._tab(out, depth + 1)
        out.write('node\n')

        for node in self.node:
            node.dump(depth + 2)

    def to_connx_text(self, path):
        # Assign ID to value_info, intializer and sparse_initializer
        for id, value_info in zip(range(1, len(self.value_info) + 1), self.value_info):
            value_info.id = id
            if value_info.initializer is not None:
                value_info.initializer.id = id
            elif value_info.sparse_initializer is not None:
                value_info.sparse_initializer.id = id

        with open(path, 'w') as out:
            # output count id id, ...
            out.write('output ')
            out.write(str(len(self.output)))
            out.write(' ')

            for output in self.output:
                out.write(str(output.id))
                out.write(' ')

            out.write('\n')

            # input count id id, ...
            out.write('input ')
            out.write(str(len(self.input)))
            out.write(' ')

            for input in self.input:
                out.write(str(input.id))
                out.write(' ')

            out.write('\n')

            # node count
            out.write('node ')
            out.write(str(len(self.node)))
            out.write('\n')

            for node in self.node:
                node.to_connx_text(out)

class ConnxTensorProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        self.id = 0

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('TensorProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        out.write(self.proto.name)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('data_type ')
        out.write(ConnxTensorProto.DataType(self.proto.data_type))
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('dims ')
        for i in range(len(self.proto.dims)):
            out.write(str(self.proto.dims[i]))
            out.write(' ')
        out.write('\n')

    def DataType(data_type):
        if data_type == 1:
            return 'FLOAT'
        elif data_type == 2:
            return 'UINT8'
        elif data_type == 3:
            return 'INT8'
        elif data_type == 4:
            return 'UINT16'
        elif data_type == 5:
            return 'INT16'
        elif data_type == 6:
            return 'INT32'
        elif data_type == 7:
            return 'INT64'
        elif data_type == 8:
            return 'STRING'
        elif data_type == 9:
            return 'BOOL'
        elif data_type == 10:
            return 'FLOAT16'
        elif data_type == 11:
            return 'DOUBLE'
        elif data_type == 12:
            return 'UINT32'
        elif data_type == 13:
            return 'UINT64'
        elif data_type == 14:
            return 'COMPLEX64'
        elif data_type == 15:
            return 'COMPLEX128'
        elif data_type == 16:
            return 'BFLOAT16'
        else:
            return 'UNDEFINED'


class ConnxSparseTensorProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        self.id = 0

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('SparseTensorProto\n')


class ConnxValueInfoProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        self.id = 0
        self.type = ConnxTypeProto(proto.type, self)
        self.initializer = None
        self.sparse_initializer = None

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('ValueInfoProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        out.write(self.proto.name)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('type\n')
        self.type.dump(depth + 2)


class ConnxTypeProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('TypeProto\n')

        if self.proto.tensor_type != None:
            self._tab(out, depth + 1)
            out.write('tensor_type\n')
            tensor_type = self.proto.tensor_type

            self._tab(out, depth + 2)
            out.write('elem_type ')
            out.write(ConnxTensorProto.DataType(tensor_type.elem_type))
            out.write('\n')

            self._tab(out, depth + 2)
            out.write('shape ')

            for i in range(len(tensor_type.shape.dim)):
                dim = tensor_type.shape.dim[i]

                if dim.dim_param != '':
                    out.write(dim.dim_param)
                else:
                    out.write(str(dim.dim_value))
                out.write(' ')

            out.write('\n')
        elif self.proto.sequence_type != None:
            out.write('WARNING: sequence type is not supported')
        elif self.proto.map_type != None:
            out.write('WARNING: map type is not supported')


class ConnxAttributeProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('AttributeProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        out.write(self.proto.name)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('type ')
        out.write(ConnxAttributeProto.AttributeType(self.proto.type))
        out.write('\n')

        self._tab(out, depth + 1)
        if self.proto.type in [ 1, 2, 3, 6, 7, 8 ]:
            out.write('value ')
            out.write(str(self.value()))
            out.write('\n')
        else:
            out.write('value\n')
            self.value().dump(depth + 2)

    def to_connx_text(self, out):
        out.write(self.proto.name)
        out.write(' ')
        out.write(str(self.proto.type))
        out.write(' ')

        if self.proto.type in [ 1, 2 ]:
            out.write(str(self.value()))
            out.write(' ')
        elif self.proto.type == 3:
            s = self.value()
            out.write(str(len(s)))
            out.write(' ')
            out.write(s.decode('utf-8')) 
            out.write(' ')
        elif self.proto.type in [ 6, 7, 8 ]:
            value = self.value()
            out.write(str(len(value)))
            out.write(' ')

            for i in range(len(value)):
                out.write(str(value[i]))
                out.write(' ')
        else:
            raise 'Not implemented yet: AttributeType ' + str(self.proto.type)

    def value(self):
        type = self.proto.type

        if type == 1:
            return self.proto.f
        elif type == 2:
            return self.proto.i
        elif type == 3:
            return self.proto.s
        elif type == 4:
            # TODO: Proto를 동적으로 만들면 안되고, value_info에 초기에 초기화 한 후 id를 할당 받아야 함. 그리고 실제 활용할 때는 id로 접근해야 함
            return ConnxTensorProto(self.proto.t, None)
        elif type == 5:
            # TODO: graph의 경우 id 또는 name으로 접근해야 함
            # value에선 당연히 숫자 또는 문자를 넘겨야 함
            return ConnxGraphProto(self.proto.g, None)
        elif type == 11:
            return ConnxSparseTensorProto(self.proto.sparse_tensor, None)
        elif type == 6:
            return self.proto.floats
        elif type == 7:
            return self.proto.ints
        elif type == 8:
            return self.proto.strings
        elif type == 9:
            return [ ConnxTensorProto(self.proto.tensors[i]) for i in range(len(self.proto.tensors)) ]
        elif type == 10:
            return [ ConnxGraphProto(self.proto.graphs[i]) for i in range(len(self.proto.graphs)) ]
        elif type == 12:
            return [ ConnxSparseTensorProto(self.proto.sparse_tensors[i]) for i in range(len(self.proto.sparse_tensors)) ]
        else:
            return None

    def AttributeType(type):
        if type == 1:
            return 'FLOAT'
        elif type == 2:
            return 'INT'
        elif type == 3:
            return 'STRING'
        elif type == 4:
            return 'TENSOR'
        elif type == 5:
            return 'GRAPH'
        elif type == 11:
            return 'SPARSE_TENSOR'
        elif type == 6:
            return 'FLOATS'
        elif type == 7:
            return 'INTS'
        elif type == 8:
            return 'STRINGS'
        elif type == 9:
            return 'TENSORS'
        elif type == 10:
            return 'GRAPHS'
        elif type == 12:
            return 'SPARSE_TENSORS'
        else:
            return 'UNDEFINED'


class ConnxNodeProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        def get_attribute_proto(name):
            for attribute_proto in proto.attribute:
                if attribute_proto.name is name:
                    return attribute_proto

            return None

        self.attribute = [ ]

        defaults = default_attribute[proto.op_type]
        for default_attr in defaults:
            new_attr = get_attribute_proto(default_attr.name)
            if new_attr is not None:
                self.attribute.append(ConnxAttributeProto(new_attr, self))
            else:
                self.attribute.append(ConnxAttributeProto(default_attr, self))

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('NodeProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        out.write(self.proto.name)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('op_type ')
        out.write(self.proto.op_type)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('input ')
        for i in range(len(self.proto.input)):
            out.write(self.proto.input[i])
            out.write(' ')
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('output ')
        for i in range(len(self.proto.output)):
            out.write(self.proto.output[i])
            out.write(' ')
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('attribute\n')
        for attribute in self.attribute:
            attribute.dump(depth + 2)
        out.write('\n')

    def to_connx_text(self, out):
        out.write(self.proto.op_type)
        out.write(' ')

        out.write(str(len(self.proto.output)))
        out.write(' ')

        out.write(str(len(self.proto.input)))
        out.write(' ')

        out.write(str(len(self.proto.attribute)))
        out.write(' ')

        for i in range(len(self.proto.output)):
            value_info = self.parent.get_value_info(self.proto.output[i])
            out.write(str(value_info.id))
            out.write(' ')

        for i in range(len(self.proto.input)):
            value_info = self.parent.get_value_info(self.proto.input[i])
            out.write(str(value_info.id))
            out.write(' ')

        for attribute in self.attribute:
            attribute.to_connx_text(out)
            out.write(' ')

        out.write('\n')
