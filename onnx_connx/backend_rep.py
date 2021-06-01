import os
from glob import glob
import numpy as np
from .opset import get_opset, get_argcount

class Graph:
    def __init__(self, backend, path, graph_id):
        self.backend = backend
        self.path = path
        self.graph_id = graph_id

        self.text = self.__read_text()
        self.initializer = [] # initializer 1 to n (initializer[0] contains value_info #1

        self.input = []
        self.output = []
        self.value_info = [ None ]

        self.__read_text()
        self.__read_initializer()

    def interprete(self, inputs):
        self._value_info(self.text[0])
        self._output(self.text[2])
        self._input(self.text[3])

        # Set initializer
        for idx in range(len(self.initializer)):
            self.value_info[idx + 1] = self.initializer[idx]

        # Set input
        for idx, id in zip(range(len(inputs)), self.input):
            self.value_info[id] = inputs[idx]

        # Execute node
        node_count = self._node(self.text[4])
        for i in range(5, 5 + node_count):
            self._exec(self.text[i])

        return [ self.value_info[id] for id in self.output ]

    def __read_text(self):
        file_path = os.path.join(self.path, '{}.text'.format(self.graph_id))
        with open(file_path, 'r') as f:
            return f.readlines()

    def __read_initializer(self):
        tokens = list(self.text[1].split(' '))
        tokens.pop(0)
        count = int(tokens.pop(0))

        for i in range(1, count + 1):
            file_path = glob(os.path.join(self.path, '{}_{}_*.data'.format(self.graph_id, i)))[0]
            data_id, data = self.__read_data(file_path)
            self.initializer.append(data)

    def __read_data(self, path):
        tokens = os.path.basename(path).strip('.data').split('_')

        tokens.pop(0)
        data_id = tokens.pop(0)
        data_type = int(tokens.pop(0))
        dim_len = int(tokens.pop(0))
        dims = [ ]
        for i in range(dim_len):
            dims.append(int(tokens.pop(0)))

        dtype = [ None, np.float32, np.uint8, np.int8, np.uint16, np.int16,
                  np.int32, np.int64, str, bool, np.float16, np.uint32, 
                  np.uint64, np.csingle, np.cdouble, None ][data_type]

        if dtype == None:
            raise Exception('Tensor data type {} is not supported yet'.format(data_type))

        with open(path, 'rb') as f:
            buf = f.read()
            return data_id, np.frombuffer(buf, dtype=dtype).reshape(dims)

    def _value_info(self, line):
        tokens = list(line.split(' '))
        tokens.pop(0)

        # Empty values with null (0th index)
        self.value_info = [ None ] * (int(tokens.pop(0)) + 1)

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
            self.input.append(int(tokens.pop(0)))

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
            name = tokens.pop(0) # drop name length
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
        op = self.backend.opset[op_type]
        if op is None:
            raise Exception('op_type {} is not supported yet'.format(op_type))

        argcount = self.backend.argcount[op_type]

        # Make argument for the operator
        # check minimum input count
        if len(input) < argcount[0]:
            raise Exception('op_type {} must have at least {} args but {}'.format(op_type, argcount[0], len(input)))

        args = [output_count]
        if argcount[1] != -1: # argcount[1] == -1 means maximum argument count will be unlimited
            for i in range(argcount[1]):
                if i < len(input):
                    args.append(self.value_info[input[i]])
                else:
                    args.append(None)
        else:
            args = args.extend([ self.value_info[id] for id in input ])

        args.extend(attribute)

        # Execute the operator
        result = op(*args)

        # Set output
        if type(result) is tuple:
            for i, id in zip(range(len(result)), output):
                self.value_info[id] = result[i]
        else:
            self.value_info[output[0]] = result

        return True

class BackendRep(object):
    def __init__(self, path):
        self.opset = None
        self.graph_count = 0

        self.text = { }
        self.data = { }

        # parse connx
        self.__parse_connx(os.path.join(path, 'model.connx'))

        # read text
        for graph_id in range(self.graph_count):
            self.text[graph_id] = Graph(self, path, graph_id)

    def __parse_connx(self, path):
        lines = self.__read_text(path)

        # check connx version
        tokens = lines[0].split(' ')
        if tokens.pop(0) != 'connx' or int(tokens.pop(0)) > 1:
            raise Exception('not supported connx version: {}'.format(lines[0].trim()))

        # parse opset_import
        tokens = lines[1].split(' ')
        specs = []
        tokens.pop(0)
        for i in range(int(tokens.pop(0))):
            domain_len = int(tokens.pop(0))
            domain = tokens.pop(0)
            while len(domain) < domain_len:
                domain += ' ' + tokens.pop(0)

            version = int(tokens.pop(0))

            specs.append({ 'domain': domain, 'version': version })

        self.opset = get_opset(specs)
        self.argcount = get_argcount(specs)

        # parse graph
        tokens = lines[2].split(' ')
        tokens.pop(0)
        self.graph_count = int(tokens.pop(0))

    def __read_text(self, path):
        with open(path, 'r') as f:
            return f.readlines()

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        graph = self.text[0]
        return graph.interprete(inputs)

