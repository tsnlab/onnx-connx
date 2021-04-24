import os
from glob import glob
import numpy as np
from .opset.opset import opset

class Graph:
    def __init__(self, name, backend, inputs):
        self.name = name
        self.backend = backend
        self.inputs = inputs

        self.input = []
        self.output = []
        self.value_info = [ None ]
        self.text = backend.text[name]

    def interprete(self):
        self._value_info(self.text[0])
        self._output(self.text[1])
        self._input(self.text[2])
        node_count = self._node(self.text[3])

        for i in range(4, 4 + node_count):
            self._exec(self.text[i])

    def _value_info(self, line):
        tokens = list(line.split(' '))
        tokens.pop(0)

        count = int(tokens.pop(0))

        self.value_info = [ None ]
        for i in range(count):
            self.value_info.append(None)

    def _output(self, line):
        tokens = list(line.split(' '))
        tokens.pop(0)

        count = int(tokens.pop(0))

        for i in range(count):
            self.output.append(int(tokens.pop(0)))

    def _input(self, line):
        tokens = list(line.split(' '))
        tokens.pop(0)

        count = int(tokens.pop(0))

        for i in range(count):
            id = int(tokens.pop(0))
            self.input.append(id)
            self.value_info[id] = self.inputs[i]

    def _node(self, line):
        tokens = list(line.split(' '))
        tokens.pop(0)

        return int(tokens.pop(0))

    def _exec(self, line):
        tokens = list(line.split(' '))

        op_type = tokens.pop(0)
        output_count = int(tokens.pop(0))
        input_count = int(tokens.pop(0))
        attribute_count = int(tokens.pop(0))

        # parse output IDs
        output = []
        for i in range(output_count):
            output.append(int(tokens.pop(0)))

        # parse input IDs
        input = []
        for i in range(input_count):
            input.append(int(tokens.pop(0)))

        # parse attribute values
        attribute = []
        for i in range(attribute_count):
            name = tokens.pop(0) # drop attribute name
            attr_type = int(tokens.pop(0))

            if attr_type == 1: # FLOAT
                value = float(tokens.pop(0))
            elif attr_type == 2: # INT
                value = int(tokens.pop(0))
            elif attr_type == 3: # STRING
                count = int(tokens.pop(0))
                value = ''
                while len(value) < count:
                    if len(value) != 0:
                        value += ' '
                    value += tokens.pop(0)
            elif attr_type == 6: # FLOATS
                count = int(tokens.pop(0))
                value = []
                for i in range(count):
                    value.append(float(tokens.pop(0)))
            elif attr_type == 7: # INTS
                count = int(tokens.pop(0))
                value = []
                for i in range(count):
                    value.append(int(tokens.pop(0)))
            elif attr_type == 8: # STRINGS
                count = int(tokens.pop(0))
                value = []
                for i in range(count):
                    count2 = int(tokens.pop(0))
                    value2 = ''

                    while len(value2) < count2:
                        if len(value2) != 0:
                            value2 += ' '
                        value2 += tokens.pop(0)

                    value.append(value2)
            else:
                raise Exception('AttributeType: {} is not supported yet'.format(attr_type))

            attribute.append(value)

        # Get Operator
        op = opset[op_type]
        if op is None:
            raise Exception('op_type {} is not supported yet'.format(op_type))

        # Make argument for the operator
        args = [ self.value_info[id] for id in input ]
        args.extend(attribute)

        # Execute the operator
        result = op(*args)

        # Set output
        if len(output) == 1:
            self.value_info[output[0]] = result
        else:
            for i, id in zip(range(len(result)), output):
                self.value_info[id] = result[i]

        return True

class Node:
    def __init__(self):
        pass

class BackendRep(object):
    def __init__(self, path):
        self.text = { }
        self.data = { }

        # read text
        for file_path in glob(os.path.join(path, '*.text')):
            name = os.path.basename(file_path).strip('.text')
            self.text[name] = self.__read_text(file_path)

        # read data
        for file_path in glob(os.path.join(path, '*.data')):
            id, self.data[name] = self.__read_data(file_path)

    def __read_text(self, path):
        with open(path, 'r') as f:
            return f.readlines()

    def __read_data(self, path):
        tokens = os.path.basename(path).strip('.data').split('_')

        tokens.pop(0) # drop name
        id = tokens.pop(0)
        data_type = int(tokens.pop(0))
        dim_len = int(tokens.pop(0))
        dims = [ ]
        for i in range(dim_len):
            dims.append(int(tokens.pop(0)))

        dtype = [ None, np.float32, np.uint8, np.int8, np.uint16, np.int16,
                  np.int32, np.int64, np.str, np.bool, np.float16, np.uint32, 
                  np.uint64, np.csingle, np.cdouble, None ][data_type]

        if dtype == None:
            raise Exception('Tensor data type {} is not supported yet'.format(data_type))

        with open(path, 'rb') as f:
            buf = f.read()
            return id, np.frombuffer(buf, dtype=dtype).reshape(dims)

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        graph = Graph('main', self, inputs)
        graph.interprete()

        return [ graph.value_info[id] for id in graph.output ]
