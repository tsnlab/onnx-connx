import numpy as np
from .util import *
from .MaxPool import MaxPool
from .Conv import Conv

# Most of the implementations are fllowd by ONNX reference implementation
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

def MatMul(A, B):
    return np.matmul(A, B)

def Relu(X):
    return np.clip(X, 0, np.inf)

def Reshape(data, shape, allowzero):
    new_shape = np.copy(shape)

    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]

    reshaped = np.reshape(data, new_shape)

    return reshaped

version = 18

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
    'Conv': None,#Conv,
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
    'MatMul': MatMul,
    'MatMulInteger': None,
    'Max': None,
    'MaxPool': MaxPool,
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
    'Relu': Relu,
    'Reshape': Reshape,
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

attrset = {
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
    'Conv': [ _string('auto_pad', 'NOTSET'), _ints('dilations', []), _int('group', 1),
                 _ints('kernel_shape', []), _ints('pads', []), _ints('strides', []) ],
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
    'MaxPool': [ _string('auto_pad', 'NOTSET'), _int('ceil_mode', 0), _ints('dilations', []), 
                 _ints('kernel_shape', []), _ints('pads', []), _int('storage_order', 0), _ints('strides', []) ],
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
    'Reshape': [ _int('allowzero', 0) ],
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
