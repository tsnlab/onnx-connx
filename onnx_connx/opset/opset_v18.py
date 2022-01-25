import onnx


def _null(name):
    attr = onnx.AttributeProto()
    attr.name = name
    attr.type = onnx.AttributeProto.AttributeType.UNDEFINED
    return attr


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
    attr.strings.extends([str.encode(s, 'utf-8') for s in value])
    return attr


version = 18

attrset = {
    '_ref': [_int('ref_count', 0)],
    'Abs': None,
    'Acos': None,
    'Acosh': None,
    'Add': [],
    'And': [],
    'ArgMax': None,
    'ArgMin': None,
    'Asin': [],
    'Asinh': None,
    'Atan': None,
    'Atanh': None,
    'AveragePool': None,
    'BatchNormalization': [_float('epsilon', 1e-05), _float('momentum', 0.9), _int('training_mode', 0)],
    'Bernoulli': None,
    'BitShift': None,
    'Cast': [_int('to', 0)],
    'CastLike': None,
    'Ceil': None,
    'Celu': None,
    'Clip': [],
    'Compress': None,
    'Concat': [_int('axis', 0)],
    'ConcatFromSequence': None,
    'Constant': None,
    'ConstantOfShape': None,
    'Conv': [_string('auto_pad', 'NOTSET'), _ints('dilations', []), _int('group', 1),
             _ints('kernel_shape', []), _ints('pads', []), _ints('strides', [])],
    'ConvInteger': None,
    'ConvTranspose': None,
    'Cos': [],
    'Cosh': [],
    'CumSum': None,
    'DepthToSpace': None,
    'DequantizeLinear': None,
    'Det': None,
    'Div': [],
    'Dropout': None,
    'DynamicQuantizeLinear': None,
    'Einsum': None,
    'Elu': None,
    'Equal': [],
    'Erf': None,
    'Exp': [],
    'Expand': None,
    'EyeLike': None,
    'Flatten': None,
    'Floor': None,
    'Function': None,
    'GRU': None,
    'Gather': [_int('axis', 0)],
    'GatherElements': None,
    'GatherND': None,
    'Gemm': None,
    'GlobalAveragePool': [],
    'GlobalLpPool': None,
    'GlobalMaxPool': [],
    'Greater': [],
    'GreaterOrEqual': [],
    'HardSigmoid': None,
    'HardSwish': None,
    'Hardmax': None,
    'Identity': [],
    'If': None,
    'InstanceNormalization': None,
    'IsInf': None,
    'IsNaN': None,
    'LRN': None,
    'LSTM': None,
    'LeakyRelu': [_float('alpha', 0.01)],
    'Less': [],
    'LessOrEqual': [],
    'Log': [],
    'LogSoftmax': None,
    'Loop': None,
    'LpNormalization': None,
    'LpPool': None,
    'MatMul': [],
    'MatMulInteger': None,
    'Max': None,
    'MaxPool': [_string('auto_pad', 'NOTSET'), _int('ceil_mode', 0), _ints('dilations', []),
                _ints('kernel_shape', []), _ints('pads', []), _int('storage_order', 0), _ints('strides', [])],
    'MaxRoiPool': None,
    'MaxUnpool': None,
    'Mean': None,
    'MeanVarianceNormalization': None,
    'Min': None,
    'Mod': None,
    'Mul': [],
    'Multinomial': None,
    'Neg': None,
    'NegativeLogLikelihoodLoss': None,
    'NonMaxSuppression': None,
    'NonZero': [],
    'Not': [],
    'OneHot': None,
    'Or': [],
    'PRelu': None,
    'Pad': None,
    'Pow': None,
    'QLinearConv': None,
    'QLinearMatMul': None,
    'QuantizeLinear': None,
    'RNN': None,
    'RandomNormal': None,
    'RandomNormalLike': None,
    'RandomUniform': None,
    'RandomUniformLike': None,
    'Range': None,
    'Reciprocal': None,
    'ReduceL1': None,
    'ReduceL2': None,
    'ReduceLogSum': None,
    'ReduceLogSumExp': None,
    'ReduceMax': None,
    'ReduceMean': None,
    'ReduceMin': None,
    'ReduceProd': None,
    'ReduceSum': None,
    'ReduceSumSquare': None,
    'Relu': [],
    'Reshape': [_int('allowzero', 0)],
    'Resize': [_string('coordinate_transformation_mode', 'half_pixel'),
               _float('cubic_coeff_a', -0.75),
               _int('exclude_outside', 0),
               _float('extrapolation_value', 0.0),
               _string('mode', 'nearest'),
               _string('nearest_mode', 'round_prefer_floor')],
    'ReverseSequence': None,
    'RoiAlign': None,
    'Round': None,
    'Scan': None,
    'Scatter (deprecated)': None,
    'ScatterElements': None,
    'ScatterND': None,
    'Selu': None,
    'SequenceAt': None,
    'SequenceConstruct': None,
    'SequenceEmpty': None,
    'SequenceErase': None,
    'SequenceInsert': None,
    'SequenceLength': None,
    'Shape': [_null('end'), _int('start', 0)],
    'Shrink': None,
    'Sigmoid': [],
    'Sign': None,
    'Sin': [],
    'Sinh': [],
    'Size': None,
    'Slice': [],
    'Softplus': [],
    'Softsign': None,
    'Softmax': None,
    'SoftmaxCrossEntropyLoss': None,
    'SpaceToDepth': None,
    'Split': [_int('axis', 0)],
    'SplitToSequence': None,
    'Sqrt': None,
    'Squeeze': [],
    'StringNormalizer': None,
    'Sub': [],
    'Sum': None,
    'Tan': [],
    'Tanh': [],
    'TfIdfVectorizer': None,
    'ThresholdedRelu': None,
    'Tile': [],
    'TopK': None,
    'Transpose': [_ints('perm', [])],
    'Trilu': None,
    'Unique': None,
    'Unsqueeze': None,
    'Upsample (deprecated)': None,
    'Where': None,
    'Xor': [],
}
