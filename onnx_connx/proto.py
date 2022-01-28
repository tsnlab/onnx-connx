import itertools
import os
import struct
import sys

import onnx
from onnx import numpy_helper

from .opset import get_attrset


class ConnxObject:
    def __init__(self, proto=None, parent=None):
        self.proto = proto
        self.parent = parent

    def _tab(self, out, depth):
        for i in range(depth):
            out.write('\t')

    def get_root(self):
        node = self

        while node.parent is not None:
            node = node.parent

        return node

    def dump(self, depth):
        pass

    def compile(self):
        pass


class ConnxModelProto(ConnxObject):
    def __init__(self, proto):
        super().__init__(proto)

        self.config = {
            'comment': True
        }
        self.next_graph_id = 0

        # Parse opset_import
        specs = []
        for i in range(len(proto.opset_import)):
            opset_import = proto.opset_import[i]
            specs.append({'domain': opset_import.domain, 'version': opset_import.version})

        self.attrset, self.verset = get_attrset(specs)

        # parse
        self.graph = ConnxGraphProto(proto.graph, self)

    def set_config(self, name, value):
        self.config[name] = value

    def get_config(self, name):
        return self.config[name]

    def alloc_graph_id(self):
        id = self.next_graph_id
        self.next_graph_id += 1

        return id

    def dump(self, depth=0):
        out = sys.stdout

        self._tab(out, depth)
        out.write('ModelProto\n')

        self._tab(out, depth + 1)
        out.write('opset_import ')
        out.write(str(len(self.proto.opset_import)))
        out.write(' ')

        for i in range(len(self.proto.opset_import)):
            opset_import = self.proto.opset_import[i]
            domain = opset_import.domain
            version = opset_import.version

            out.write(str(len(domain)))
            out.write(' ')
            out.write(domain)
            out.write(' ')
            out.write(str(version))
            out.write('\n')

        self.graph.dump(depth + 1)

    def compile(self, path):
        os.makedirs(path, exist_ok=True)

        # Write graph
        self.graph.compile(path)

        with open(os.path.join(path, 'model.connx'), 'w') as out:
            # Write connx version
            out.write('connx 5\n')

            # Write opset_import
            out.write('opset_import ')
            out.write(str(len(self.proto.opset_import)))
            out.write(' ')

            for i in range(len(self.proto.opset_import)):
                opset_import = self.proto.opset_import[i]
                domain = opset_import.domain
                version = opset_import.version

                out.write(str(len(domain)))
                out.write(' ')
                out.write(domain)
                out.write(' ')
                out.write(str(version))
                out.write('\n')

            # Write number of graphs
            out.write('graph ')
            out.write(str(self.next_graph_id))
            out.write('\n')


class ConnxGraphProto(ConnxObject):
    def __init__(self, proto, parent):
        super().__init__(proto, parent)

        self.id = self.get_root().alloc_graph_id()

        self.initializer = [ConnxTensorProto(proto, self) for proto in proto.initializer]
        self.sparse_initializer = [ConnxSparseTensorProto(proto, self) for proto in proto.sparse_initializer]
        self.input = []
        self.output = []
        self.value_info = [ConnxValueInfoProto(proto, self) for proto in proto.value_info]

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
            if input.name == '':
                continue

            value_info = self.get_value_info(input.name)

            if value_info is None:
                value_info = ConnxValueInfoProto(input, self)
                self.value_info.append(value_info)

            self.input.append(value_info)

        # Make value_info reflects to output
        for output in proto.output:
            if output.name == '':
                continue

            value_info = self.get_value_info(output.name)

            if value_info is None:
                value_info = ConnxValueInfoProto(output, self)
                self.value_info.append(value_info)

            self.output.append(value_info)

        # Make value_info reflects to node
        for node in proto.node:
            for name in itertools.chain(node.input, node.output):
                value_info = self.get_value_info(name)

                if name == '':
                    continue

                if value_info is None:
                    value_info = ConnxValueInfoProto(None, self, name=name)
                    self.value_info.append(value_info)

        # Assign ID to value_info, which has intializer and sparse_initializer first
        id = 1
        for value_info in self.value_info:
            if value_info.initializer is not None:
                value_info.id = id
                value_info.initializer.id = id
                id += 1
            elif value_info.sparse_initializer is not None:
                value_info.id = id
                value_info.sparse_initializer.id = id
                id += 1

        # Assign ID to value_info, which doesn't have intializer and sparse_initializer next
        for value_info in self.value_info:
            if value_info.initializer is None and value_info.sparse_initializer is None:
                value_info.id = id
                id += 1

        self.node = [ConnxNodeProto(proto, self) for proto in proto.node]

        # Internal operators
        for initializer in self.initializer:
            idx = 0
            ref_count = self.get_ref_count(initializer.proto.name)
            if ref_count > 1:
                node = self.make_ref_count(initializer.proto.name, initializer.id, ref_count)
                self.node.insert(idx, node)
                idx += 1

        for sparse_initializer in self.sparse_initializer:
            idx = 0
            ref_count = self.get_ref_count(sparse_initializer.proto.name)
            if ref_count > 1:
                node = self.make_ref_count(sparse_initializer.proto.name, sparse_initializer.id, ref_count)
                self.node.insert(idx, node)
                idx += 1

        idx = 0
        length = len(self.node)
        while idx < length:
            node = self.node[idx]
            for name in node.proto.output:
                ref_count = self.get_ref_count(name, idx + 1)
                if ref_count > 1:
                    value_info = self.get_value_info(name)
                    node = self.make_ref_count(name, value_info.id, ref_count)
                    self.node.insert(idx + 1, node)
                    idx += 1
                    length += 1
            idx += 1

    def get_value_info(self, name):
        for value_info in self.value_info:
            if value_info.proto is not None and value_info.proto.name == name or value_info.name == name:
                return value_info

        return None

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('GraphProto\n')

        self._tab(out, depth + 1)
        out.write('id ')
        out.write(str(self.id))
        out.write('\n')

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

    def get_ref_count(self, name, idx=0):
        count = 0

        for i in range(idx, len(self.node)):
            if name in self.node[i].proto.input:
                count += 1

        if name in self.proto.output:
            count += 1

        return count

    def make_ref_count(self, name, id, ref_count):
        attr = onnx.AttributeProto()
        attr.name = 'ref_count'
        attr.type = onnx.AttributeProto.AttributeType.INT
        attr.i = ref_count

        proto = onnx.NodeProto()
        proto.name = '_ref'
        proto.op_type = '_ref'
        proto.input.append(name)
        proto.attribute.append(attr)

        return ConnxNodeProto(proto, self)

    def compile(self, path):
        def write(tensor, array, out):
            out.write(struct.pack('=I', tensor.proto.data_type))  # dtype
            out.write(struct.pack('=I', len(tensor.proto.dims)))  # ndim

            for i in range(len(tensor.proto.dims)):  # dims
                out.write(struct.pack('=I', tensor.proto.dims[i]))

            buf = array.tobytes()
            out.write(buf)

        for initializer in self.initializer:
            name = '{}_{}.data'.format(str(self.id), str(initializer.id))

            with open(os.path.join(path, name), 'wb') as out:
                array = numpy_helper.to_array(initializer.proto)
                write(initializer, array, out)

        for sparse_initializer in self.sparse_initializer:
            name = '{}_{}.data'.format(str(self.id), str(sparse_initializer.id))

            # TODO: Compress array
            with open(os.path.join(path, name), 'wb') as out:
                array = numpy_helper.to_array(sparse_initializer.proto)
                write(initializer, array, out)

        # graph name
        with open(os.path.join(path, str(self.id) + '.text'), 'w') as out:
            # variable count
            out.write('value_info ')
            out.write(str(len(self.value_info)))
            out.write('\n')

            # initializer count
            out.write('initializer ')
            out.write(str(len(self.initializer) + len(self.sparse_initializer)))
            out.write('\n')

            # output count id id, ...
            out.write('output ')
            out.write(str(len(self.output)))

            for i in range(len(self.output)):
                out.write(' ')
                out.write(str(self.output[i].id))

            out.write('\n')

            # input count id id, ...
            out.write('input ')
            out.write(str(len(self.input)))

            for i in range(len(self.input)):
                out.write(' ')
                out.write(str(self.input[i].id))

            out.write('\n')

            # node count
            out.write('node ')
            out.write(str(len(self.node)))
            out.write('\n')

            for node in self.node:
                node.compile(out)


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
        out.write('dims')
        for i in range(len(self.proto.dims)):
            out.write(' ')
            out.write(str(self.proto.dims[i]))
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
    def __init__(self, proto, parent, name=None):
        super().__init__(proto, parent)

        self.id = 0
        self.type = ConnxTypeProto(proto.type, self) if proto is not None else None
        self.name = name
        self.initializer = None
        self.sparse_initializer = None

    def dump(self, depth):
        out = sys.stdout

        self._tab(out, depth)
        out.write('ValueInfoProto\n')

        self._tab(out, depth + 1)
        out.write('name ')
        if self.proto is not None:
            out.write(self.proto.name)
        else:
            out.write('<generated>')
        out.write('\n')

        if self.type is not None:
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

        if self.proto.tensor_type is not None:
            self._tab(out, depth + 1)
            out.write('tensor_type\n')
            tensor_type = self.proto.tensor_type

            self._tab(out, depth + 2)
            out.write('elem_type ')
            out.write(ConnxTensorProto.DataType(tensor_type.elem_type))
            out.write('\n')

            self._tab(out, depth + 2)
            out.write('shape')

            for i in range(len(tensor_type.shape.dim)):
                dim = tensor_type.shape.dim[i]

                out.write(' ')
                if dim.dim_param != '':
                    out.write(dim.dim_param)
                else:
                    out.write(str(dim.dim_value))

            out.write('\n')
        elif self.proto.sequence_type is not None:
            out.write('WARNING: sequence type is not supported')
        elif self.proto.map_type is not None:
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
        if self.proto.type in [1, 2, 3, 6, 7, 8]:
            out.write('value ')
            out.write(str(self.value()))
            out.write('\n')
        else:
            out.write('value\n')
            self.value().dump(depth + 2)

    def compile(self, out):
        out.write(str(self.proto.type))

        if self.proto.type == 0:
            pass  # UNDEFINED means null attribute
        elif self.proto.type in [1, 2]:
            out.write(' ')
            out.write(str(self.value()))
        elif self.proto.type == 3:
            out.write(' ')
            s = self.value().decode('utf-8')
            out.write(str(len(s)))
            out.write(' ')
            out.write(s)
        elif self.proto.type in [6, 7]:
            out.write(' ')
            value = self.value()
            out.write(str(len(value)))

            for i in range(len(value)):
                out.write(' ')
                out.write(str(value[i]))
        elif self.proto.type == 8:
            out.write(' ')
            value = self.value().decode('utf-8')
            out.write(str(len(value)))

            for i in range(len(value)):
                out.write(' ')
                out.write(len(value[i]))
                out.write(' ')
                out.write(value[i])
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
            return [ConnxTensorProto(self.proto.tensors[i]) for i in range(len(self.proto.tensors))]
        elif type == 10:
            return [ConnxGraphProto(self.proto.graphs[i]) for i in range(len(self.proto.graphs))]
        elif type == 12:
            return [ConnxSparseTensorProto(self.proto.sparse_tensors[i]) for i in range(len(self.proto.sparse_tensors))]
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
                if attribute_proto.name == name:
                    return attribute_proto

            return None

        root = self.get_root()

        self.version = root.verset[proto.op_type]
        self.attribute = []

        attrset = root.attrset[proto.op_type]

        if attrset is None:
            raise Exception(f'op_type {proto.op_type} is not supported yet')

        for default_attr in attrset:
            original_attr = get_attribute_proto(default_attr.name)
            if original_attr is not None:
                self.attribute.append(ConnxAttributeProto(original_attr, self))
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
        out.write('_')
        out.write(str(self.version))
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('input')
        for i in range(len(self.proto.input)):
            out.write(' ')
            out.write(self.proto.input[i])
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('output')
        for i in range(len(self.proto.output)):
            out.write(' ')
            out.write(self.proto.output[i])
        out.write('\n')

        self._tab(out, depth + 1)
        out.write('attribute\n')
        for attribute in self.attribute:
            attribute.dump(depth + 2)
        out.write('\n')

    def compile(self, out):
        out.write(self.proto.op_type)
        out.write('_')
        out.write(str(self.version))
        out.write(' ')

        out.write(str(len(self.proto.output)))
        out.write(' ')

        out.write(str(len(self.proto.input)))
        out.write(' ')

        out.write(str(len(self.attribute)))

        for i in range(len(self.proto.output)):
            value_info = self.parent.get_value_info(self.proto.output[i])
            out.write(' ')
            if value_info is not None:
                out.write(str(value_info.id))
            else:
                out.write('0')

        for i in range(len(self.proto.input)):
            value_info = self.parent.get_value_info(self.proto.input[i])
            out.write(' ')
            if value_info is not None:
                out.write(str(value_info.id))
            else:
                out.write('0')

        for attribute in self.attribute:
            out.write(' ')
            attribute.compile(out)

        if self.get_root().get_config('comment'):
            out.write(' # ')
            out.write(str(len(self.proto.name)))
            out.write(' ')
            out.write(self.proto.name)

            for i in range(len(self.proto.output)):
                name = self.proto.output[i]
                out.write(' ')
                out.write(str(len(name)))
                out.write(' ')
                out.write(name)

            for i in range(len(self.proto.input)):
                name = self.proto.input[i]
                out.write(' ')
                out.write(str(len(name)))
                out.write(' ')
                out.write(name)

            for attribute in self.attribute:
                name = attribute.proto.name
                out.write(' ')
                out.write(str(len(name)))
                out.write(' ')
                out.write(name)

        out.write('\n')
