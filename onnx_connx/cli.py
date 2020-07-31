import itertools
import argparse
import sys
import os
import onnx
from onnx import numpy_helper
import numpy as np

is_output_comment = True
encoding = 'little'

def get_output_dir(path):
    if path == None:
        path = 'out'

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Cannot make output directory:', path)
        return None

    return path

def normalize_attributes(op, attrs):
    def attr_int(name, value):
        attr = onnx.AttributeProto()
        attr.name = name
        attr.type = onnx.AttributeProto.INT
        attr.i = value

        return attr

    def attr_float(name, value):
        attr = onnx.AttributeProto()
        attr.name = name
        attr.type = onnx.AttributeProto.FLOAT
        attr.f = value

        return attr

    def attr_string(name, value):
        attr = onnx.AttributeProto()
        attr.name = name
        attr.type = onnx.AttributeProto.STRING
        attr.s = value

        return attr

    def attr_ints(name, value):
        attr = onnx.AttributeProto()
        attr.name = name
        attr.type = onnx.AttributeProto.INTS

        if type(value) is int:
            for i in range(len(value)):
                attr.ints.append(0)
        else:
            for v in value:
                attr.ints.append(v)

        return attr

    def attr_floats(name, value):
        attr = onnx.AttributeProto()
        attr.name = name
        attr.type = onnx.AttributeProto.FLOATS

        if type(value) is int:
            for i in range(len(value)):
                attr.floats.append(0.0)
        else:
            for v in value:
                attr.floats.append(v)

        return attr

    def attr_strings(name, value):
        attr = onnx.AttributeProto()
        attr.name = name
        attr.type = onnx.AttributeProto.STRINGS

        if type(value) is int:
            for i in range(len(value)):
                attr.strings.append('')
        else:
            for v in value:
                attr.strings.append(v)

        return attr

    def get_attr(name):
        for attr in attrs:
            if attr.name == name:
                return attr

        return None

    result = []

    if op == 'Conv':
        kernel_shape = get_attr('kernel_shape')

        result.append(get_attr('auto_pad') or attr_string('auto_pad', 'NOTSET'))
        result.append(get_attr('dilations') or attr_ints('dilations', np.ones(len(kernel_shape.ints), np.int32)))
        result.append(get_attr('group') or attr_int('group', 1))
        result.append(kernel_shape)
        result.append(get_attr('pads') or attr_ints('pads', np.zeros(len(kernel_shape.ints) * 2, np.int32)))
        result.append(get_attr('strides') or attr_ints('strides', np.zeros(len(kernel_shape.ints) * 2, np.int32)))
    elif op == 'MaxPool':
        kernel_shape = get_attr('kernel_shape')

        result.append(get_attr('auto_pad') or attr_string('auto_pad', 'NOTSET'))
        result.append(get_attr('ceil_mode') or attr_int('ceil_mode', 0))
        result.append(get_attr('dilations') or attr_ints('dilations', np.ones(len(kernel_shape.ints), np.int32)))
        result.append(kernel_shape)
        result.append(get_attr('pads') or attr_ints('pads', np.zeros(len(kernel_shape.ints) * 2, np.int32)))
        result.append(get_attr('storage_order') or attr_int('storage_order', 0))
        result.append(get_attr('strides') or attr_ints('strides', np.zeros(len(kernel_shape.ints) * 2, np.int32)))

    return result

class Value:
    def __init__(self):
        self.type = None
        self.name = None
        self.id = None
        self.proto = None

class Call:
    def __init__(self):
        self.outputs = []
        self.inputs = []
        self.attributes = []
        self.proto = None

class Path:
    def __init__(self, parent):
        self.parent = parent 
        self.id = None
        self.outputs = []
        self.inputs = []
        self.output_paths = []
        self.input_paths = []
        self.calls = []

    def dump(self, output_dir, file):
        file.write('path ' + str(self.id) + '\n')
        file.write('\tinput_paths ' + ' '.join(str(path.id) for path in self.input_paths) + '\n')
        file.write('\toutput_paths ' + ' '.join(str(path.id) for path in self.output_paths) + '\n')

        global is_output_comment

        if is_output_comment:
            file.write('\n\t# inputs: ' + str(len(self.inputs)) + '\n')

        if len(self.inputs) > 0:
            for id, name in zip(self.calls[0].inputs, self.calls[0].proto.input):
                file.write('\tinput ' + str(id) + ' ')

                if is_output_comment:
                    file.write('# ' + name)

                file.write('\n')

        if is_output_comment:
            file.write('\n\t# calls: ' + str(len(self.calls)) + '\n')

        for i in range(len(self.calls)):
            call = self.calls[i]

            attrs = normalize_attributes(call.proto.op_type, call.proto.attribute)

            file.write('\tcall ')
            file.write(str(call.proto.op_type) + ' ')
            file.write(str(len(call.outputs)) + ' ')
            file.write(str(len(call.inputs)) + ' ')
            file.write(str(len(attrs)) + ' ')

            for output in call.outputs:
                file.write(str(output) + ' ')

            for input in call.inputs:
                file.write(str(input) + ' ')

            for attr in attrs:
                file.write(str(self.parent.parent.alloc_attribute(attr)) + ' ')

            if is_output_comment:
                file.write('# ')

                file.write(' '.join(call.proto.output) + ' ')
                file.write(' '.join(call.proto.input) + ' ')

                for attr in attrs:
                    file.write(attr.name + '=' + self.comment_attribute(attr) + ' ')

            file.write('\n')

            # garbage collection
            for output in call.outputs:
                if not self.is_use_after(output, i + 1):
                    file.write('\tdelete ' + str(output) + '\n')
                    self.parent.deleted.append(output)

            for input in call.inputs:
                if not self.is_use_after(input, i + 1):
                    file.write('\tdelete ' + str(input) + '\n')
                    self.parent.deleted.append(input)

        if is_output_comment:
            file.write('\n\t# outputs: ' + str(len(self.outputs)) + '\n')

        if len(self.outputs) > 0 and len(self.calls) > 0:
            for id, name in zip(self.calls[-1].outputs, self.calls[-1].proto.output):
                file.write('\toutput ' + str(id) + ' ')

                if is_output_comment:
                    file.write('# ' + name)

                file.write('\n')

        file.write('\n')

    def comment_attribute(self, attr):
        comment = ''

        if attr.type == onnx.AttributeProto.FLOAT:
            comment += str(attr.f)
        elif attr.type == onnx.AttributeProto.INT:
            comment += str(attr.i)
        elif attr.type == onnx.AttributeProto.STRING:
            comment += '"' + attr.s.decode('utf-8') + '"'
        elif attr.type == onnx.AttributeProto.TENSOR:
            comment += '<tensor>'
        elif attr.type == onnx.AttributeProto.GRAPH:
            comment += '<graph>'
        elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
            comment += '<sparse_tensor>'
        elif attr.type == onnx.AttributeProto.FLOATS:
            comment += '['
            length = len(attr.floats)
            for i in range(length):
                comment += str(attr.floats[i])
                if i + 1 < length:
                    comment += ','
            comment += ']'
        elif attr.type == onnx.AttributeProto.INTS:
            comment += '['
            length = len(attr.ints)
            for i in range(length):
                comment += str(attr.ints[i])
                if i + 1 < length:
                    comment += ','
            comment += ']'
        elif attr.type == onnx.AttributeProto.STRINGS:
            comment += '['
            length = len(attr.strings)
            for i in range(length):
                comment += '"' + attr.strings[i].decode('utf-8') + '"'
                if i + 1 < length:
                    comment += ','
            comment += ']'
        elif attr.type == onnx.AttributeProto.TENSORS:
            comment += '<tensors>'
        elif attr.type == onnx.AttributeProto.GRAPHS:
            comment += '<graphs>'
        elif attr.type == onnx.AttributeProto.SPARSE_TENSORS:
            comment += '<sparse_tensors>'
        else:
            raise Exception('Illegal attribute type: ' + str(attr.type))

        return comment

    def is_use_after(self, id, idx):
        for i in range(idx, len(self.calls)):
            call = self.calls[i]

            if id in call.outputs or id in call.inputs:
                return True

        if len(self.calls) > 0 and id in self.calls[-1].outputs:
            return True

        for path in self.output_paths:
            if path.is_use_after(id, 0):
                return True

        return False

class Graph:
    def __init__(self, parent, model):
        self.parent = parent
        self.inputs = []
        self.outputs = []
        self.values = {}
        self.initializer_names = []
        self.paths = []
        self.deleted = []
        self.proto = model

        # add initializers to values
        for initializer in model.initializer:
            value = Value()
            value.type = 'tensor'
            value.name = initializer.name
            value.id = len(self.values)
            value.proto = initializer

            self.values[value.name] = value
            self.initializer_names.append(value.name)

        for sparse_initializer in model.sparse_initializer:
            value = Value()
            value.type = 'sparse_tensor'
            value.name = sparse_initializer.name
            value.id = len(self.values)
            value.proto = sparse_initializer

            self.values[value.name] = value
            self.initializer_names.append(value.name)

        '''Algorithm description

        Graph.inputPath is virtual input path
        Graph.outputPath is virtual output path
        unresolved is a queue
        resolved is a queue
        push(unresolved, outputPath)
        push(resolved, inputPath)
        while(len(unresolved) > 0)
            path = peek(unresolved)
            if(path.inputNameCount == 0)
                pop(unresolved)
                push(resolved, path)
                continue

            nodes, paths = find_dependencies(path)

            if(not found)
                exception
            
            if(len(nodes) == 1 and len(paths) == 0)
                extend path
            else 
                for node in nodes
                    push(unresolved, newPath(node))

                add new paths and old paths to the path

                pop(unresolved)
                push(resolved, path)
        '''

        # set graph's inputs and outputs
        for input in model.input:
            value = self.get_value(input.name)
            self.inputs.append(value.id)

        for output in model.output:
            value = self.get_value(output.name)
            self.outputs.append(value.id)

        # make call list
        calls = []
        for node in model.node:
            call = Call()

            for output in node.output:
                value = self.get_value(output)
                call.outputs.append(value.id)

            for input in node.input:
                value = self.get_value(input)
                call.inputs.append(value.id)

            call.attributes.extend(node.attribute)
            call.proto = node

            calls.append(call)

        """
        @return (list of calls, list of paths)
        """
        def find_dependencies(inputs):
            found = []

            dep_calls = []
            dep_paths = []

            for call in calls:
                for input in inputs:
                    if input in found:
                        continue

                    if input in call.proto.output:
                        dep_calls.append(call)
                        found.append(input)

            for p in itertools.chain(resolved, unresolved):
                for input in inputs:
                    if input in found:
                        continue

                    if input in p.outputs:
                        dep_paths.append(p)
                        found.append(input)

            for input in inputs:
                if input in found:
                    continue

                if input in self.initializer_names:
                    found.append(input)

            if len(found) != len(inputs):
                raise Exception('Cannot resolve input: ' + ', '.join(filter(found.__ne__, inputs)))
            else:
                return dep_calls, dep_paths

        # init working queue and done queue
        unresolved = []
        resolved = []

        input_path = Path(self)
        for input in model.input:
            if not input.name in self.initializer_names:
                input_path.outputs.append(input.name)
        resolved.append(input_path)

        output_path = Path(self)
        for output in model.output:
            output_path.inputs.append(output.name)
            output_path.outputs.append(output.name)
        unresolved.append(output_path)

        # loop queue
        while len(unresolved) > 0:
            path = unresolved[0]

            if len(path.inputs) == 0:
                del unresolved[0]
                resolved.append(path)
                continue

            dep_calls, dep_paths = find_dependencies(path.inputs)

            if len(dep_calls) == 1 and len(dep_paths) == 0:
                calls.remove(dep_calls[0])
                path.calls.insert(0, dep_calls[0])
                path.inputs = list(dep_calls[0].proto.input)
            else:
                for dep_call in dep_calls:
                    new_path = Path(self)
                    new_path.outputs.extend(dep_call.proto.output)
                    new_path.inputs.extend(dep_call.proto.input)
                    new_path.calls.append(dep_call)

                    path.input_paths.append(new_path)
                    new_path.output_paths.append(path)
                    calls.remove(dep_call)

                    unresolved.append(new_path)

                for dep_path in dep_paths:
                    if not dep_path in path.input_paths:
                        path.input_paths.append(dep_path)

                    if not path in dep_path.output_paths:
                        dep_path.output_paths.append(path)

                del unresolved[0]
                resolved.append(path)

        self.paths.append(resolved[0])
        self.paths.extend(reversed(resolved[1:]))
        for i in range(len(self.paths)):
            self.paths[i].id = i

    def get_value(self, name):
        if name in self.values:
            return self.values[name]
        else:
            value = Value()
            value.type = 'none'
            value.name = name
            value.id = len(self.values)

            self.values[value.name] = value

            return value

    def dump(self, output_dir, file):
        global is_output_comment

        if is_output_comment:
            file.write('# initializers\n')

        if len(self.proto.initializer) > 0:
            file.write('tensor 0 ' + str(len(self.proto.initializer) - 1))

            if is_output_comment:
                file.write(' # ')

                for initializer in self.proto.initializer:
                    file.write(initializer.name + ' ')

            file.write('\n')

        if len(self.proto.sparse_initializer) > 0:
            file.write('sparse_tensor ' + str(len(self.proto.initializer)) + ' ' + str(len(self.proto.initializer) + len(self.proto.sparse_initializer) - 1))

            if is_output_comment:
                file.write(' # ')

                for initializer in self.proto.sparse_initializer:
                    file.write(initializer.name + ' ')

            file.write('\n')

        for initializer in self.initializer_names:
            value = self.values[initializer]
            if value.type == 'tensor':
                name = str(value.id) + '_tensor_' + str(value.proto.data_type)
                for dim in value.proto.dims:
                    name += '_' + str(dim)
                name += '.np'

                array = numpy_helper.to_array(value.proto)
                array.tofile(output_dir + os.path.sep + name)
            else:
                raise Exception('value type ' + value.type + ' is not supported yet')

        if is_output_comment:
            file.write('\n# variables\n')

        if len(self.values) > 0:
            file.write('variable ' + str(len(self.initializer_names)) + ' ' + str(len(self.values) - 1))

            if is_output_comment:
                file.write(' # ')

                for key, value in self.values.items():
                    if value.name in self.initializer_names:
                        continue

                    file.write(value.name + ' ')

            file.write('\n')

        file.write('\n')

        if is_output_comment:
            file.write('\n# paths\n')

        for path in self.paths:
            path.dump(output_dir, file)

        if is_output_comment:
            file.write('\n# run\n')

        file.write('start ')
        for path in self.paths:
            if len(path.input_paths) == 0:
                file.write(str(path.id) + ' ')

        file.write('\n')

        file.write('stop ')
        for path in self.paths:
            if len(path.output_paths) == 0:
                file.write(str(path.id) + ' ')

        file.write('\n')

        for key, value in self.values.items():
            if not value.id in self.deleted:
                file.write('clean ' + str(value.id) + '\n')

class Model():
    def __init__(self, model):
        self.proto = model
        self.graph = Graph(self, model.graph)
        self.attributes = []

    def dump(self, output_dir):
        global is_output_comment

        file = open(output_dir + os.path.sep + 'main.cos', 'w')

        if is_output_comment:
            file.write('# metadata\n')

        file.write('opset ')
        for opset in self.proto.opset_import:
            file.write(str(opset.version) + ' ')

        file.write('\n\n')

        self.graph.dump(output_dir, file)

        file.close()

        self.dump_attributes(output_dir)

    def has_attribute(self, attr):
        for i in range(len(self.attributes)):
            attr2 = self.attributes[i]

            if attr.type != attr2.type:
                continue

            if attr.type == onnx.AttributeProto.FLOAT:
                if attr.f == attr2.f:
                    return i
            elif attr.type == onnx.AttributeProto.INT:
                if attr.i == attr2.i:
                    return i
            elif attr.type == onnx.AttributeProto.STRING:
                if attr.s == attr2.s:
                    return i
            elif attr.type == onnx.AttributeProto.TENSOR:
                array = numpy_helper.to_array(attr.t)
                array2 = numpy_helper.to_array(attr2.t)

                if attr == attr2:
                    return i
            elif attr.type == onnx.AttributeProto.GRAPH:
                break
            elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                raise Exception('sparse_tensor type is not supported yet')
            elif attr.type == onnx.AttributeProto.FLOATS:
                if attr.floats == attr2.floats:
                    return i
            elif attr.type == onnx.AttributeProto.INTS:
                if attr.ints == attr2.ints:
                    return i
            elif attr.type == onnx.AttributeProto.STRINGS:
                if attr.strings == attr2.strings:
                    return i
            elif attr.type == onnx.AttributeProto.TENSORS:
                length = len(attr.tensors)
                if length != len(attr2.tensors):
                    continue

                for j in range(length):
                    array = numpy_helper.to_array(attr.tensors[j])
                    array2 = numpy_helper.to_array(attr2.tensors[j])

                    if attr != attr2:
                        continue

                return i
            elif attr.type == onnx.AttributeProto.GRAPHS:
                break
            elif attr.type == onnx.AttributeProto.SPARSE_TENSORS:
                raise Exception('sparse_tensors type is not supported yet')
            else:
                raise Exception('Illegal attribute type: ' + str(attr.type))

        return -1

    def alloc_attribute(self, attr):
        id = self.has_attribute(attr)
        if id >= 0:
            return id

        id = len(self.attributes)
        self.attributes.append(attr)
        return id

    def dump_attributes(self, output_dir):
        index = []

        # write data
        with open(output_dir + os.path.sep + 'attr.db', 'wb') as file:
            align = 4
            offset = 0

            def write(buf):
                nonlocal offset
                nonlocal file

                offset += file.write(buf)

                pad = (align - (offset % align)) % align
                if pad > 0:
                    offset += file.write(np.zeros(pad, np.uint8).tobytes())

            for attr in self.attributes:
                
                if attr.type == onnx.AttributeProto.FLOAT:
                    index.append(offset)
                    write(np.array([ attr.f ], np.float32).tobytes())
                elif attr.type == onnx.AttributeProto.INT:
                    index.append(offset)
                    write(np.array([ attr.i ], np.int32).tobytes())
                elif attr.type == onnx.AttributeProto.STRING:
                    index.append(offset)

                    # write length
                    length = len(attr.s)
                    write(np.array([ length ], np.uint32).tobytes())

                    # write string
                    write(attr.s)
                elif attr.type == onnx.AttributeProto.TENSOR:
                    index.append(offset)

                    # write type
                    write(np.array([ attr.t.type ], np.uint32).tobytes())

                    # write dimension
                    dimension = len(attr.t.dims)
                    write(np.array([ dimension ], np.uint32).tobytes())

                    # write lengths
                    for i in range(dimension):
                        write(np.array([ attr.t.dims[i] ], np.uint32).tobytes())

                    # write array
                    array = numpy_helper.to_array(attr.t)
                    write(array)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    raise Exception('graph writing is not supported yet')
                elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                    raise Exception('sparse_tensor writing is not supported yet')
                elif attr.type == onnx.AttributeProto.FLOATS:
                    index.append(offset)

                    # write length
                    write(np.array([ len(attr.floats) ], np.uint32).tobytes())

                    # write array
                    write(np.array(attr.floats, np.float32).tobytes())
                elif attr.type == onnx.AttributeProto.INTS:
                    index.append(offset)

                    # write length
                    write(np.array([ len(attr.ints) ], np.uint32).tobytes())

                    # write array
                    write(np.array(attr.ints, np.int32).tobytes())
                elif attr.type == onnx.AttributeProto.STRINGS:
                    index.append(offset)

                    # write length
                    write(np.array([ len(attr.strings) ], np.uint32).tobytes())

                    for s in attr.strings:
                        # write length
                        length = len(s)
                        write(np.array([ length ], np.uint32).tobytes())

                        # write string
                        write(s)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    raise Exception('tensors writing is not supported yet')
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    raise Exception('graph writing is not supported yet')
                elif attr.type == onnx.AttributeProto.SPARSE_TENSORS:
                    raise Exception('sparse_tensors writing is not supported yet')
                else:
                    raise Exception('Illegal attribute type: ' + str(attr.type))

        # write index
        np.array(index, np.uint32).tofile(output_dir + os.path.sep + 'attr.idx')

    def encode_attr(self, attr):
        text = str(attr.type) + ' '
        id = len(self.attrributes)
        self.attributes.append(attr)
        text += str(id)

        return text

def main():
    parser = argparse.ArgumentParser(description='ONNX-CONNX Command Line Interface')
    parser.add_argument('onnx', metavar='onnx', type=argparse.FileType('rb'), nargs=1, help='an input ONNX model')
    parser.add_argument('-o', metavar='connx', type=str, nargs='?', help='an output CONNX model directory')
    parser.add_argument('-c', metavar='comment', type=str, nargs='?', choices=['true', 'false', 'True', 'False'], help='output comments(true or false)')

    # parse args
    args = parser.parse_args()

    # comment option
    global is_output_comment
    is_output_comment = args.c == None or args.c == 'true' or args.c == 'True'

    # mkdir output_dir
    output_dir = get_output_dir(args.o)
    if output_dir == None:
        return

    # parse onnx file
    onnx_model = None
    try:
        onnx_model = onnx.load_model(args.onnx[0])
    except:
        print('Illegal ONNX file format:', args.onnx[0].name)
        return

    
    model = Model(onnx_model)
    model.dump(output_dir)

if __name__ == '__main__':
    main()
