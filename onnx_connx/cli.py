from __future__ import annotations
import os
import glob
import argparse
import itertools
from typing import List

import numpy as np
import onnx
from onnx import numpy_helper
from .normalizer import normalize
from .config import Config

config = Config()


def encode_tensor(tensor):
    buf = bytearray()

    # write type
    buf += np.array([tensor.data_type], np.uint32).tobytes()

    # write dimension
    dimension = len(tensor.dims)
    buf += np.array([dimension], np.uint32).tobytes()

    # write lengths
    for i in range(dimension):
        buf += np.array([tensor.dims[i]], np.uint32).tobytes()

    # write array
    buf += numpy_helper.to_array(tensor).tobytes()

    return buf


class Value:
    def __init__(self):
        self.type = None
        self.name = None
        self.id = None
        self.ref_count = 0
        self.proto = None


class Call:
    def __init__(self):
        self.output_calls = []
        self.input_calls = []

        self.outputs = []
        self.inputs = []
        self.attributes = []

        self.proto = None

    def insert_before(self, call):
        # replace self.input_calls.output_calls = call
        for input_call in self.input_calls:
            for i in range(len(input_call.output_calls)):
                if input_call.output_calls[i] == self:
                    input_call.output_calls[i] = call

        # set call.input_calls = self.input_calls
        call.input_calls = self.input_calls

        # set call.output_calls = self
        call.output_calls.append(self)

        # set self.input_call = call
        self.input_calls = [call]

    def insert_after(self, call):
        # replace self.output_calls.input_calls = call
        for output_call in self.output_calls:
            for i in range(len(output_call.input_calls)):
                if output_call.input_calls[i] == self:
                    output_call.input_calls[i] = call

        # set call.output_calls = self.output_calls
        call.output_calls = self.output_calls

        # set call.input_calls = self
        call.input_calls.append(self)

        # set self.output_call = call
        self.output_calls = [call]

    def find_gc_call(self, call_id: int, ref_count: int) -> Call:
        """
        Find garbage collection call for value id
        :param call_id: value id which is created by the call
        :param ref_count: reference count of the value
        :return: gc call which has responsible to delete the value
        """

        paths = [[next_call] for next_call in self.output_calls]
        ref_counts = [0] * len(paths)
        visited = {}

        # make paths
        i = 0
        while i < len(paths):
            path = paths[i]
            last_call = path[-1]

            if call_id in last_call.inputs:
                ref_counts[i] += 1

            if last_call in visited:  # don't extend but copy already extended one
                org_path = visited[last_call]
                idx = org_path.index(last_call)
                path.extend(org_path[idx + 1:])
                i += 1
                continue

            if (len(last_call.input_calls) > 1 or len(
                    last_call.output_calls) > 1) and last_call not in visited:  # Check the node is branch
                visited[last_call] = path

            if len(last_call.output_calls) == 0:  # end of graph
                i += 1
            else:
                for j in range(len(last_call.output_calls)):
                    next_call = last_call.output_calls[j]

                    if j == 0:
                        next_path = path
                    else:
                        next_path = []
                        paths.append(next_path)
                        ref_counts.append(0)

                    next_path.append(next_call)

        # prune paths which is not using the value
        candidates = []
        candidate_ref_counts = []
        for i in range(len(paths)):
            if ref_counts[i] > 0:
                candidates.append(paths[i])
                candidate_ref_counts.append(ref_counts[i])

        if len(candidates) == 0:
            raise Exception('There is no GC path for value: ' + str(call_id) + '(# of paths: ' + str(len(paths)) + ')')

        # find gc call
        if len(candidates) == 1:  # the last one of last input
            for call2 in candidates[0]:
                if call_id in call2.inputs:
                    ref_count -= 1

                if ref_count == 0:
                    return call2
        else:  # the last common call
            length = min((len(path) for path in candidates))

            last_common_call = None
            path = candidates[0]
            for i in range(length):
                call2 = path[-(i + 1)]
                for p in candidates[1:]:
                    if call2 != p[-(i + 1)]:
                        break

                last_common_call = call2

            if last_common_call is not None:
                return last_common_call

        raise Exception(
            'Cannot find GC call for value: ' + str(call_id) + ' (# of candidate paths: ' + str(len(candidates)) + ')')

    def name(self):
        if self.proto is None:
            if len(self.output_calls) != 0 and len(self.input_calls) == 0:
                return 'input ' + str(self.outputs)
            elif len(self.output_calls) == 0 and len(self.input_calls) != 0:
                return 'output ' + str(self.inputs)
            else:
                return 'delete ' + str(self.inputs)
        else:
            return self.proto.name + ':' + self.proto.op_type

    def print(self):
        print('call ', self.name(), 'inputs:', self.inputs, 'outputs:', self.outputs)
        print('\t', 'input_calls:', ' '.join([call.name() for call in self.input_calls]))
        print('\t', 'output_calls:', ' '.join([call.name() for call in self.output_calls]))

    def dump(self, file):
        file.write('\tcall ')

        if self.proto is None:
            if len(self.output_calls) != 0 and len(self.input_calls) == 0:
                file.write('input ')
            elif len(self.output_calls) == 0 and len(self.input_calls) != 0:
                file.write('output ')
            else:
                file.write('delete ')
        else:
            file.write(str(self.proto.op_type) + ' ')

        file.write(str(len(self.outputs)) + ' ')
        file.write(str(len(self.inputs)) + ' ')
        file.write(str(len(self.attributes)) + '  ')

        for output_id in self.outputs:
            file.write(str(output_id) + ' ')

        file.write(' ')

        for input_id in self.inputs:
            file.write(str(input_id) + ' ')

        file.write(' ')

        for attr_id in self.attributes:
            file.write(str(attr_id) + ' ')

        if config.output_comment and self.proto is not None:
            file.write('# ')

            file.write(' '.join(self.proto.output) + ' ')
            file.write(' '.join(self.proto.input) + ' ')

            for attr in self.proto.attribute:
                file.write(attr.name + '=' + self.comment_attribute(attr) + ' ')

        file.write('\n')

    def comment_attribute(self, attr):
        comment = ''

        if attr.type == onnx.AttributeProto.FLOAT:
            comment += str(attr.f)
        elif attr.type == onnx.AttributeProto.INT:
            comment += str(attr.i)
        elif attr.type == onnx.AttributeProto.STRING:
            comment += '"' + attr.s.decode() + '"'
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
                comment += '"' + attr.strings[i].decode() + '"'
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


class Path:
    def __init__(self):
        self.id = None
        self.output_paths = []
        self.input_paths = []
        self.calls = []

    def fill(self, call):
        while call is not None:
            self.calls.append(call)

            if len(call.output_calls) == 1 and len(call.output_calls[0].input_calls) == 1:
                call = call.output_calls[0]
            else:
                call = None

    def fill_backword(self, call):
        while call is not None:
            self.calls.insert(0, call)

            if len(call.input_calls) == 1 and len(call.input_calls[0].output_calls) == 1:
                call = call.input_calls[0]
            else:
                call = None

    def dump(self, file):
        file.write('path ' + str(self.id) + '\n')
        file.write('\tinput_paths ' + ' '.join(str(path.id) for path in self.input_paths) + '\n')
        file.write('\toutput_paths ' + ' '.join(str(path.id) for path in self.output_paths) + '\n')

        # calls
        file.write('\n\tcalls ' + str(len(self.calls)) + '\n')

        for i in range(len(self.calls)):
            call = self.calls[i]
            call.dump(file)

        file.write('\n')


class Graph:
    def __init__(self, parent, model):
        self.parent = parent
        self.inputs = []
        self.outputs = []
        self.values = {}
        self.values_by_id = []
        self.initializer_names = []
        self.paths = []
        self.proto = model

    def init(self):
        # add null to values
        value = Value()
        value.type = 'null'
        value.name = 'null'
        value.id = len(self.values)
        value.proto = None

        self.values[value.name] = value
        self.values_by_id.append(value)
        self.initializer_names.append(value.name)

        # add tensor initializers to values 
        for initializer in self.proto.initializer:
            value = Value()
            value.type = 'tensor'
            value.name = initializer.name
            value.id = len(self.values)
            value.proto = initializer

            self.values[value.name] = value
            self.values_by_id.append(value)
            self.initializer_names.append(value.name)

        # add sparse_tensor initializers to values 
        for sparse_initializer in self.proto.sparse_initializer:
            value = Value()
            value.type = 'sparse_tensor'
            value.name = sparse_initializer.name
            value.id = len(self.values)
            value.proto = sparse_initializer

            self.values[value.name] = value
            self.values_by_id.append(value)
            self.initializer_names.append(value.name)

        # set graph's inputs and outputs
        for input_proto in self.proto.input:
            if input_proto.name in self.initializer_names:
                continue

            value = self.get_value(input_proto.name)
            self.inputs.append(value.id)

        for output_proto in self.proto.output:
            value = self.get_value(output_proto.name)
            self.outputs.append(value.id)

        self.schedule()

    def schedule(self):
        # make call tree
        input_call = Call()
        input_call.outputs.extend(self.inputs)

        output_call = Call()
        output_call.inputs.extend(self.outputs)
        for output_id in self.outputs:
            value = self.values_by_id[output_id]
            value.ref_count += 1

        calls = [input_call, output_call]

        for node in self.proto.node:
            call = Call()

            for output_name in node.output:
                value = self.get_value(output_name)
                call.outputs.append(value.id)

            for input_name in node.input:
                value = self.get_value(input_name)
                value.ref_count += 1
                call.inputs.append(value.id)

            attrs = normalize(node.op_type, node.attribute)
            for attr in attrs:
                call.attributes.append(self.parent.alloc_attribute(attr))

            call.proto = node

            calls.append(call)

        # make dependency (call.input_calls, output_calls)
        for call in calls:
            input_missing: List[int] = [*call.inputs]
            output_missing: List[int] = [*call.outputs]

            # find from initializers
            input_missing[:] = itertools.filterfalse(lambda _id: _id < len(self.initializer_names), input_missing)
            output_missing[:] = itertools.filterfalse(lambda _id: _id < len(self.initializer_names), output_missing)

            # find from calls
            for call2 in calls:
                if call == call2:
                    continue

                # check input dependency
                old_input_len = len(input_missing)
                input_missing[:] = itertools.filterfalse(lambda _id: _id in call2.outputs, input_missing)

                if old_input_len != len(input_missing):
                    if call not in call2.output_calls:
                        call2.output_calls.append(call)

                    if call2 not in call.input_calls:
                        call.input_calls.append(call2)

                # check output dependency
                old_output_len = len(output_missing)
                output_missing[:] = itertools.filterfalse(lambda _id: _id in call2.inputs, output_missing)

                if old_output_len != len(output_missing):
                    if call2 not in call.output_calls:
                        call.output_calls.append(call2)

                    if call not in call2.input_calls:
                        call2.input_calls.append(call)

            # Check missing dependency
            if len(input_missing) > 0:
                raise Exception('Cannot find input dependency for call ' + call.name() + ': ' + str(input_missing))

            if len(output_missing) > 0:
                for orphan_id in output_missing:
                    print('WARN: orphan output:', orphan_id, self.values_by_id[orphan_id].name)

        # make dependency (input_call -> call input only initializers)
        for call in calls[1:]:  # ignore input_call itself
            if len(call.input_calls) == 0 and call not in input_call.output_calls:
                input_call.output_calls.append(call)
                call.input_calls.append(input_call)

        # garbage collection
        for call in calls[2:]:  # ignore input_call and output_call
            for call_id in call.outputs:
                if call_id in self.inputs or call_id in self.outputs:  # ignore input values and output values
                    continue

                value = self.values_by_id[call_id]
                if value.ref_count == 0:  # ignore orphan value
                    continue

                gc_call = call.find_gc_call(call_id, value.ref_count)

                if gc_call is not None:
                    next_call = None
                    if len(gc_call.output_calls) == 1:
                        next_call = gc_call.output_calls[0]

                    if gc_call.proto is None and gc_call is not input_call and gc_call is not output_call:
                        # merge deletes
                        gc_call.inputs.append(call_id)
                    elif next_call is not None and next_call.proto is None and next_call is not input_call and \
                            next_call is not output_call:
                        # merge deletes
                        next_call.inputs.append(call_id)
                    else:
                        # insert delete
                        delete_call = Call()
                        delete_call.inputs.append(call_id)
                        gc_call.insert_after(delete_call)

        # make paths
        paths = {}  # key: start call, value: path

        input_path = Path()
        input_path.fill(input_call)
        paths[input_call] = input_path

        output_path = Path()
        output_path.fill_backword(output_call)
        paths[output_path.calls[0]] = output_path

        unresolved = [input_path]

        # make path's input/output relationship
        while len(unresolved) > 0:
            path = unresolved[0]

            for next_call in path.calls[-1].output_calls:
                if next_call in paths:
                    next_path = paths[next_call]
                else:
                    next_path = Path()
                    next_path.fill(next_call)

                    unresolved.append(next_path)
                    paths[next_call] = next_path

                path.output_paths.append(next_path)
                next_path.input_paths.append(path)

            del unresolved[0]

        # order paths
        input_path.id = len(self.paths)
        self.paths.append(input_path)
        unresolved = [*input_path.output_paths]

        while len(unresolved) > 0:
            path = unresolved[0]

            is_resolved = True

            for prev_path in path.input_paths:
                if prev_path not in self.paths:
                    is_resolved = False
                    break

            del unresolved[0]

            if is_resolved:
                path.id = len(self.paths)
                self.paths.append(path)

                for next_path in path.output_paths:
                    if next_path not in self.paths and next_path not in unresolved:
                        unresolved.append(next_path)
            else:
                unresolved.append(path)

    def get_value(self, name):
        if name in self.values:
            return self.values[name]
        else:
            value = Value()
            value.type = 'none'
            value.name = name
            value.id = len(self.values)

            self.values[value.name] = value
            self.values_by_id.append(value)

            return value

    def dump(self, file):
        global config

        file.write('initializers ' + str(1 + len(self.proto.initializer) + len(self.proto.sparse_initializer)))
        if config.output_comment:
            file.write(' # ')

            file.write('null ')

            for initializer in self.proto.initializer:
                file.write(initializer.name + ' ')

            for initializer in self.proto.sparse_initializer:
                file.write(initializer.name + ' ')

        file.write('\n')

        # write initializer.db
        index = [0]
        with open(config.output_path + os.path.sep + 'init.db', 'wb') as fp:
            for initializer in self.initializer_names:
                value = self.values[initializer]
                if value.type == 'null':
                    length = fp.write(np.array([0], np.uint32).tobytes())
                    index.append(index[-1] + length)
                elif value.type == 'tensor':
                    length = fp.write(encode_tensor(value.proto))
                    index.append(index[-1] + length)
                else:
                    raise Exception('value type ' + value.type + ' is not supported yet')

        del index[-1]

        # write initializer.idx
        with open(config.output_path + os.path.sep + 'init.idx', 'wb') as fp:
            for offset in index:
                fp.write(np.array([offset], np.uint32).tobytes())

        del index

        # variables
        if len(self.values) > 0:
            file.write('variables ' + str(len(self.values) - len(self.initializer_names)))

            if config.output_comment:
                file.write(' # ')

                for key, value in self.values.items():
                    if value.name in self.initializer_names:
                        continue

                    file.write(value.name + ' ')

            file.write('\n')

        file.write('\n')

        # paths
        file.write('paths ')
        file.write(str(len(self.paths)))
        file.write('\n')

        for path in self.paths:
            path.dump(file)

        if config.output_comment:
            file.write('# run\n')

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


class Model:
    def __init__(self, model):
        self.proto = model
        self.graph = Graph(self, model.graph)
        self.attributes = []

        null_attr = onnx.AttributeProto()
        null_attr.name = 'null'
        null_attr.type = onnx.AttributeProto.UNDEFINED
        self.attributes.append(null_attr)

        self.graph.init()

    def dump(self):
        global config

        file = open(config.output_path + os.path.sep + 'main.cnx', 'w')

        file.write('opset ')
        opset_ver = -1
        for opset in self.proto.opset_import:
            if opset.version > opset_ver:
                opset_ver = opset.version
        file.write(str(opset_ver) + ' ')
        file.write('\n')

        self.graph.dump(file)

        file.close()

        self.dump_attributes()

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

                if array == array2:
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

                    if array != array2:
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
        if attr.type == onnx.AttributeProto.UNDEFINED:
            return 0

        _id = self.has_attribute(attr)
        if _id >= 0:
            return _id

        _id = len(self.attributes)
        self.attributes.append(attr)

        return _id

    def dump_attributes(self):
        global config

        index = []

        # write data
        with open(config.output_path + os.path.sep + 'attr.db', 'wb') as file:
            offset = 0

            def write(_buf, _size):
                nonlocal offset
                nonlocal file

                pad = config.alignof(offset, _size)
                if pad > 0:
                    offset += file.write(np.zeros(pad, np.uint8).tobytes())

                offset += file.write(_buf)

                pad = config.padof(offset, _size)
                if pad > 0:
                    offset += file.write(np.zeros(pad, np.uint8).tobytes())

            for attr in self.attributes:

                if attr.type == onnx.AttributeProto.UNDEFINED:
                    index.append(offset)
                    write(np.array([0], np.uint32).tobytes(), 4)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    index.append(offset)
                    write(np.array([attr.f], np.float32).tobytes(), 4)
                elif attr.type == onnx.AttributeProto.INT:
                    index.append(offset)
                    write(np.array([attr.i], np.int32).tobytes(), 4)
                elif attr.type == onnx.AttributeProto.STRING:
                    index.append(offset)

                    # write string
                    buf = bytearray(attr.s)
                    buf.append(0)
                    write(buf, 1)
                elif attr.type == onnx.AttributeProto.TENSOR:
                    index.append(offset)

                    buf = encode_tensor(attr.t)
                    write(buf, 0)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    raise Exception('graph writing is not supported yet')
                elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                    raise Exception('sparse_tensor writing is not supported yet')
                elif attr.type == onnx.AttributeProto.FLOATS:
                    index.append(offset)

                    # write length
                    write(np.array([len(attr.floats)], np.uint32).tobytes(), 4)

                    # write array
                    write(np.array(attr.floats, np.float32).tobytes(), 4)
                elif attr.type == onnx.AttributeProto.INTS:
                    index.append(offset)

                    # write length
                    write(np.array([len(attr.ints)], np.uint32).tobytes(), 4)

                    # write array
                    write(np.array(attr.ints, np.int32).tobytes(), 4)
                elif attr.type == onnx.AttributeProto.STRINGS:
                    index.append(offset)

                    # write length
                    write(np.array([len(attr.strings)], np.uint32).tobytes(), 4)

                    sub_offset = 4 + len(attr.string) * 4
                    sub_offsets = []
                    sub_buf = []

                    align = 4

                    for s in attr.strings:
                        # write length
                        buf = bytearray(s)
                        buf.append(0)

                        sub_offsets.append(sub_offset)
                        sub_buf.append(buf)
                        sub_offset += len(buf)
                        sub_offset += (align - (sub_offset % align)) % align

                    # write index
                    for off in sub_offsets:
                        write(np.array([off], np.uint32).tobytes(), 4)

                    # write strings
                    for buf in sub_buf:
                        write(buf, 1)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    raise Exception('tensors writing is not supported yet')
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    raise Exception('graph writing is not supported yet')
                elif attr.type == onnx.AttributeProto.SPARSE_TENSORS:
                    raise Exception('sparse_tensors writing is not supported yet')
                else:
                    raise Exception('Illegal attribute type: ' + str(attr.type))

        # write index
        np.array(index, np.uint32).tofile(config.output_path + os.path.sep + 'attr.idx')


def main(*_args: str) -> object:
    parser = argparse.ArgumentParser(description='ONNX-CONNX Command Line Interface')
    parser.add_argument('onnx', metavar='onnx or pb', nargs='+', help='an input ONNX model')
    parser.add_argument('-p', metavar='profile', type=str, nargs='?', help='specify configuration file')
    parser.add_argument('-o', metavar='output', type=str, nargs='?', help='output directory(default is out)')
    parser.add_argument('-c', metavar='comment', type=str, nargs='?', choices=['true', 'false', 'True', 'False'],
                        help='output comments(true or false)')

    # parse args
    if len(_args) > 0:
        args = parser.parse_args(_args)
    else:
        args = parser.parse_args()

    inputs = []
    for path in args.onnx:
        for p in glob.glob(path):
            inputs.append(open(p, 'rb'))

    # config
    global config

    if args.p is not None:
        config.read(args.p)

    if args.o is not None:
        config.set_output_path(args.o)

    if args.c is not None:
        config.set_output_comment(args.c)

    config.make_output_path()

    # read onnx and pb
    for file in inputs:
        # parse onnx file
        if file.name.endswith('.onnx'):
            try:
                onnx_model = onnx.load_model(file)
            except ValueError:
                print('Illegal ONNX file format:', file.name)
                return

            model = Model(onnx_model)
            model.dump()
        # parse pb file
        elif file.name.endswith('.pb'):
            try:
                tensor_model = onnx.load_tensor(file)
            except ValueError:
                print('Illegal protocol buffer format:', file.name)
                return

            buf = encode_tensor(tensor_model)
            basename = os.path.basename(file.name)
            with open(config.output_path + os.path.sep + basename[:-3] + '.tensor', 'wb') as fp:
                fp.write(buf)
        else:
            print('Not supported file format:', file.name)
            return


if __name__ == '__main__':
    main()
