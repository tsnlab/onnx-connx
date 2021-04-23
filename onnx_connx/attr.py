import onnx

def _float(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.FLOAT
    attr.f = value
    return attr

def _int(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.INT
    attr.i = value
    return attr

def _string(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.STRING
    attr.s = str.encode(value, 'utf-8')
    return attr

def _floats(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.FLOATS
    attr.floats.extend(value)
    return attr

def _ints(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.INTS
    attr.ints.extend(value)
    return attr

def _strings(name, value):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.STRINGS
    attr.strings.extends([ str.encode(s, 'utf-8') for s in value ])
    return attr

default_attribute = {
    'Abs': [ ],
    'Acos': [ ],
    'Acosh': [ ],
    'Add': [ ],
    'And': [ ],
    'ArgMax': [ _int('axis', 0), _int('keepdims', 1), _int('select_last_index', 0) ],
    'ArgMin': [ _int('axis', 0), _int('keepdims', 1), _int('select_last_index', 0) ],
    'Asin': [ ],
    'Asinh': [ ],
    'Atan': [ ],
    'Atanh': [ ],
    'AveragePool': [ _string('auto_pad', 'NOTSET'), _int('ceil_mode', 0), _int('count_include_pad', 0), _ints('kernel_shape', []), _ints('pads', []), _ints('strides', []) ],
    'BatchNormalization': [ _float('epsilon', 1e-05), _float('momentum', 0.9), _int('training_mode', 0) ],
    'BitShift': [ _string('direction', '') ],
    'Cast': [ _int('to', 0) ],
    'Ceil': [ ],
    'Celu': [ _float('alpha', 1.0) ],
    'Clip': [ ],
    'Compress': [ _int('axis', 0) ],
    'Concat': [ _int('axis', 0) ],
    'ConcatFromSequence': [ _int('axis', 0), _int('new_axis', 0) ],

    'Constant': [ ],
    'ConstantOfShape': [ ],
    'Conv': [ ],
    'ConvInteger': [ ],
    'ConvTranspose': [ ],
    'Cos': [ ],
    'Cosh': [ ],
    'CumSum': [ ],
    'DepthToSpace': [ ],
    'DequantizeLinear': [ ],
    'Det': [ ],
    'Div': [ ],
    'Dropout': [ ],
    'Einsum': [ ],
    'Elu': [ ],
    'Equal': [ ],
    'Erf': [ ],
    'Exp': [ ],
    'Expand': [ ],
    'EyeLike': [ ],
    'Flatten': [ ],
    'Floor': [ ],
    'GRU': [ ],
    'Gather': [ ],
    'GatherElements': [ ],
    'GatherND': [ ],
    'Gemm': [ ],
    'GlobalAveragePool': [ ],
    'GlobalLpPool': [ ],
    'GlobalMaxPool': [ ],
    'Greater': [ ],
    'HardSigmoid': [ ],
    'Hardmax': [ ],
    'Identity': [ ],
    'If': [ ],
    'InstanceNormalization': [ ],
    'IsInf': [ ],
    'IsNaN': [ ],
    'LRN': [ ],
    'LSTM': [ ],
    'LeakyRelu': [ ],
    'Less': [ ],
    'Log': [ ],
    'Loop': [ ],
    'LpNormalization': [ ],
    'LpPool': [ ],
    'MatMul': [ ],
    'MatMulInteger': [ ],
    'Max': [ ],
    'MaxPool': [ ],
    'MaxRoiPool': [ ],
    'MaxUnpool': [ ],
    'Mean': [ ],
    'Min': [ ],
    'Mod': [ ],
    'Mul': [ ],
    'Multinomial': [ ],
    'Neg': [ ],
    'NonMaxSuppression': [ ],
    'NonZero': [ ],
    'Not': [ ],
    'OneHot': [ ],
    'Or': [ ],
    'PRelu': [ ],
    'Pad': [ ],
    'Pow': [ ],
    'QLinearConv': [ ],
    'QLinearMatMul': [ ],
    'QuantizeLinear': [ ],
    'RNN': [ ],
    'RandomNormal': [ ],
    'RandomNormalLike': [ ],
    'RandomUniform': [ ],
    'RandomUniformLike': [ ],
    'Reciprocal': [ ],
    'ReduceL1': [ ],
    'ReduceL2': [ ],
    'ReduceLogSum': [ ],
    'ReduceLogSumExp': [ ],
    'ReduceMax': [ ],
    'ReduceMean': [ ],
    'ReduceMin': [ ],
    'ReduceProd': [ ],
    'ReduceSum': [ ],
    'ReduceSumSquare': [ ],
    'Relu': [ ],
    'Reshape': [ ],
    'Resize': [ ],
    'ReverseSequence': [ ],
    'RoiAlign': [ ],
    'Round': [ ],
    'Scan': [ ],
    'Scatter (deprecated)': [ ],
    'ScatterElements': [ ],
    'ScatterND': [ ],
    'Selu': [ ],
    'SequenceAt': [ ],
    'SequenceConstruct': [ ],
    'SequenceEmpty': [ ],
    'SequenceErase': [ ],
    'SequenceInsert': [ ],
    'SequenceLength': [ ],
    'Shape': [ ],
    'Shrink': [ ],
    'Sigmoid': [ ],
    'Sign': [ ],
    'Sin': [ ],
    'Sinh': [ ],
    'Size': [ ],
    'Slice': [ ],
    'Softplus': [ ],
    'Softsign': [ ],
    'SpaceToDepth': [ ],
    'Split': [ ],
    'SplitToSequence': [ ],
    'Sqrt': [ ],
    'Squeeze': [ ],
    'StringNormalizer': [ ],
    'Sub': [ ],
    'Sum': [ ],
    'Tan': [ ],
    'Tanh': [ ],
    'TfIdfVectorizer': [ ],
    'ThresholdedRelu': [ ],
    'Tile': [ ],
    'TopK': [ ],
    'Transpose': [ ],
    'Trilu': [ ],
    'Unique': [ ],
    'Unsqueeze': [ ],
    'Upsample (deprecated)': [ ],
    'Where': [ ],
    'Xor': [ ],
    'Function': [ ],
    'Celu': [ ],
    'DynamicQuantizeLinear': [ ],
    'GreaterOrEqual': [ ],
    'HardSwish': [ ],
    'LessOrEqual': [ ],
    'LogSoftmax': [ ],
    'MeanVarianceNormalization': [ ],
    'NegativeLogLikelihoodLoss': [ ],
    'Range': [ ],
    'Softmax': [ ],
    'SoftmaxCrossEntropyLoss': [ ],
}
