import itertools
import argparse
import sys
import os
import onnx
from onnx import numpy_helper
from numpy import tofile

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

            file.write('\tcall ')
            file.write(str(call.proto.op_type) + ' ')
            file.write(str(len(call.outputs)) + ' ')
            file.write(str(len(call.inputs)) + ' ')
            file.write(str(len(call.attributes)) + ' ')

            for output in call.outputs:
                file.write(str(output) + ' ')

            for input in call.inputs:
                file.write(str(input) + ' ')

            for attr in call.attribute:
                if attr.

            if is_output_comment:
                file.write('# ')

                file.write(' '.join(call.proto.output) + ' ')
                file.write(' '.join(call.proto.input) + ' ')

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

    def dump_attributes(self, output_dir):
        with open(output_dir + os.path.sep + 'attr.np', 'wb') as file:
            # write index

            # write data
            for attr in self.attributes:
                if attr.type == 1:  # FLOAT = 1
                    text += attr.f
                elif attr.type == 2: # INT = 2
                    text += attr.i
                elif attr.type == 3: # STRING = 3
                    text += attr.s
                elif attr.type == 4: # TENSOR = 4
                elif attr.type == 5: # GRAPH = 5
                elif attr.type == 11: # SPARSE_TENSOR = 11
                elif attr.type == 6: # FLOATS = 6
                    text += str(len(attr.floats)) + ' '
                    for value in attr.floats:
                elif attr.type == 7: # INTS = 7
                elif attr.type == 8: # STRINGS = 8
                elif attr.type == 9: # TENSORS = 9
                elif attr.type == 10: # GRAPHS = 10
                elif attr.type == 12: # SPARSE_TENSORS = 12
                else:
                    raise Exception('Illegal attribute type: ' + str(attr.type))

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
