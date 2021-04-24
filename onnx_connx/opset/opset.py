import numpy as np

def Abs(X):
    return np.abs(X)

def Acos(X):
    return np.arccos(X)

def Acosh(input):
    return np.arccosh(input)

def Add(A, B):
    return A + B

def And(A, B):
    return np.logical_and(A, B)

def ArgMax(data, axis, keepdims, select_last_index):
    if select_last_index == 1:
         data = np.flip(data, axis)

    result = np.argmax(data, axis=axis)
    if keepdims == 1:
        result = np.expand_dims(result, axis)

    if select_last_index == 1:
        result = data.shape[axis] - result - 1

    return result.astype(np.int64)

def ArgMin(data, axis, keepdims, select_last_index):
    if select_last_index == 1:
         data = np.flip(data, axis)

    result = np.argmin(data, axis=axis)
    if keepdims == 1:
        result = np.expand_dims(result, axis)

    if select_last_index == 1:
        result = data.shape[axis] - result - 1

    return result.astype(np.int64)

opset = {
    'Abs': Abs,
    'Acos': Acos,
    'Acosh': Acosh,
    'Add': Add,
    'And': And,
    'ArgMax': ArgMax,
    'ArgMin': ArgMin,
    'Asin': None,
    'Asinh': None,
    'Atan': None,
    'Atanh': None,
    'AveragePool': None,
    'BatchNormalization': None,
    'BitShift': None,
    'Cast': None,
    'Ceil': None,
    'Celu': None,
    'Clip': None,
    'Compress': None,
    'Concat': None,
    'ConcatFromSequence': None,
    'Constant': None,
    'ConstantOfShape': None,
    'Conv': None,
    'ConvInteger': None,
    'ConvTranspose': None,
    'Cos': None,
    'Cosh': None,
    'CumSum': None,
    'DepthToSpace': None,
    'DequantizeLinear': None,
    'Det': None,
    'Div': None,
    'Dropout': None,
    'Einsum': None,
    'Elu': None,
    'Equal': None,
    'Erf': None,
    'Exp': None,
    'Expand': None,
    'EyeLike': None,
    'Flatten': None,
    'Floor': None,
    'GRU': None,
    'Gather': None,
    'GatherElements': None,
    'GatherND': None,
    'Gemm': None,
    'GlobalAveragePool': None,
    'GlobalLpPool': None,
    'GlobalMaxPool': None,
    'Greater': None,
    'HardSigmoid': None,
    'Hardmax': None,
    'Identity': None,
    'If': None,
    'InstanceNormalization': None,
    'IsInf': None,
    'IsNaN': None,
    'LRN': None,
    'LSTM': None,
    'LeakyRelu': None,
    'Less': None,
    'Log': None,
    'Loop': None,
    'LpNormalization': None,
    'LpPool': None,
    'MatMul': None,
    'MatMulInteger': None,
    'Max': None,
    'MaxPool': None,
    'MaxRoiPool': None,
    'MaxUnpool': None,
    'Mean': None,
    'Min': None,
    'Mod': None,
    'Mul': None,
    'Multinomial': None,
    'Neg': None,
    'NonMaxSuppression': None,
    'NonZero': None,
    'Not': None,
    'OneHot': None,
    'Or': None,
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
    'Relu': None,
    'Reshape': None,
    'Resize': None,
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
    'Shape': None,
    'Shrink': None,
    'Sigmoid': None,
    'Sign': None,
    'Sin': None,
    'Sinh': None,
    'Size': None,
    'Slice': None,
    'Softplus': None,
    'Softsign': None,
    'SpaceToDepth': None,
    'Split': None,
    'SplitToSequence': None,
    'Sqrt': None,
    'Squeeze': None,
    'StringNormalizer': None,
    'Sub': None,
    'Sum': None,
    'Tan': None,
    'Tanh': None,
    'TfIdfVectorizer': None,
    'ThresholdedRelu': None,
    'Tile': None,
    'TopK': None,
    'Transpose': None,
    'Trilu': None,
    'Unique': None,
    'Unsqueeze': None,
    'Upsample (deprecated)': None,
    'Where': None,
    'Xor': None,
    'Function': None,
    'Celu': None,
    'DynamicQuantizeLinear': None,
    'GreaterOrEqual': None,
    'HardSwish': None,
    'LessOrEqual': None,
    'LogSoftmax': None,
    'MeanVarianceNormalization': None,
    'NegativeLogLikelihoodLoss': None,
    'Range': None,
    'Softmax': None,
    'SoftmaxCrossEntropyLoss': None,
}
