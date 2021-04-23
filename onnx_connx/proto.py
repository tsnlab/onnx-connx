class ConnxObject:
    def __init__(self, proto=None, parent=None):
        self.proto = proto
        self.parent = parent

    def _tab(self, out, depth):
        for i in range(depth):
            out.write('\t')

    def dump(self, out, depth):
        pass


class ConnxModelProto(ConnxObject):
    def __init__(self, proto):
        super().__init__(proto)

        self.graph = ConnxGraphProto(proto.graph, self)

    def dump(self, out, depth):
        self._tab(out, depth)
        out.write('ModelProto\n')

        self.graph.dump(out, depth + 1)


class ConnxGraphProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        self.initializer = [ ConnxTensorProto(proto, self) for proto in proto.initializer ]
        self.sparse_initializer = [ ConnxSparseTensorProto(proto, self) for proto in proto.sparse_initializer ]

        self.input = [ ConnxValueInfoProto(proto, self) for proto in proto.input ]
        self.output = [ ConnxValueInfoProto(proto, self) for proto in proto.output ]
        self.value_info = [ ConnxValueInfoProto(proto, self) for proto in proto.value_info ]

        self.node = [ ConnxNodeProto(proto, self) for proto in proto.node ]

    def dump(self, out, depth):
        self._tab(out, depth)
        out.write('GraphProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        out.write(self.proto.name)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('initializer\n')

        for initializer in self.initializer:
            initializer.dump(out, depth + 2)

        self._tab(out, depth + 1)
        out.write('sparse_initializer\n')

        for sparse_initializer in self.sparse_initializer:
            sparse_initializer.dump(out, depth + 2)

        self._tab(out, depth + 1)
        out.write('input\n')

        for input in self.input:
            input.dump(out, depth + 2)

        self._tab(out, depth + 1)
        out.write('output\n')

        for output in self.output:
            output.dump(out, depth + 2)

        self._tab(out, depth + 1)
        out.write('node\n')

        for node in self.node:
            node.dump(out, depth + 2)


class ConnxTensorProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

    def dump(self, out, depth):
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

    def dump(self, out, depth):
        self._tab(out, depth)
        out.write('SparseTensorProto\n')


class ConnxValueInfoProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        self.type = ConnxTypeProto(proto.type, self)

    def dump(self, out, depth):
        self._tab(out, depth)
        out.write('ValueInfoProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        out.write(self.proto.name)
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('type\n')
        self.type.dump(out, depth + 2)


class ConnxTypeProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

    def dump(self, out, depth):
        self._tab(out, depth)
        out.write('TypeProto\n')

        self._tab(out, depth + 1)
        out.write('value\n')
        if self.proto.tensor_type != None:
            tensor_type = self.proto.tensor_type

            self._tab(out, depth + 2)
            out.write('elem_type ')
            out.write(ConnxTensorProto.DataType(tensor_type.elem_type))
            out.write('\n')

            self._tab(out, depth + 2)
            out.write('shape ')
            out.write(str(len(tensor_type.shape.dim)))

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

    def dump(self, out, depth):
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
            self.value().dump(out, depth + 2)
        

    def value(self):
        type = self.proto.type

        if type == 1:
            return self.proto.f
        elif type == 2:
            return self.proto.i
        elif type == 3:
            return self.proto.s
        elif type == 4:
            return ConnxTensorProto(self.proto.t, None)
        elif type == 5:
            return ConnxTensorProto(self.proto.g, None)
        elif type == 11:
            return ConnxTensorProto(self.proto.sparse_tensor, None)
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

        self.attribute = [ ConnxAttributeProto(proto, self) for proto in proto.attribute ]

    def dump(self, out, depth):
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
            attribute.dump(out, depth + 2)

