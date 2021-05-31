import numpy as np
from .util import *
from .MaxPool import MaxPool
from .BatchNormalization import BatchNormalization
from .GlobalAveragePool import GlobalAveragePool
from .Conv import Conv
from .Cast import Cast
from .Resize import Resize


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


def Concat(*inputs):
    r"""
    inputs type constraints
        tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), 
        tensor(int8), tensor(int16), tensor(int32), tensor(int64), 
        tensor(bfloat16), tensor(float16), tensor(float), tensor(double), 
        tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
    """
    axis = inputs[-1]
    inputs = inputs[:-1]
    concat_result = np.concatenate(inputs, axis)
        
    return concat_result


def Exp(input):
    r"""
    input type constraints
        tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
    """
    return np.exp(input)


def Gather(data, indices, axis):
    r"""
    :param data: Tensor of rank r >= 1.
    :param indices: Tensor of int32/int64 indices of any rank q. 
                    All index values are expected to be  
                    within bounds [-s, s-1] along axis of size s.  
                    It is an error if any of the index values are out of bounds.
    """
    return np.take(data,indices, axis)


def LeakyRelu(X, alpha):
    return np.clip(X, 0, np.inf) + np.clip(X, -np.inf, 0) * alpha


def Log(input):
    return np.log(input)


def Mul(A, B):
    return A * B


def Shape(data):
    return np.array(data.shape)


def Sigmoid(X):
    return 1.0 / (1.0 + np.exp(np.negative(X)))


def Slice(data, starts, ends, axes, steps):
    # TODO : Naive implementation.
    expr = [":"] * len(data.shape)
    if axes is None:
        axes = np.array([i for i in range(len(data.shape))])
    
    for i in range(axes.shape[0]):
        if steps is None:
            expr[axes[i]] = f"{starts[i]}:{ends[i]}"
        else:
            expr[axes[i]] = f"{starts[i]}:{ends[i]}:{steps[i]}"

    slice_str = (",").join(expr)
    eval_str = f"data[{slice_str}]"
    
    return eval(eval_str)


def Split(input, split, axis):
    print(input.shape, type(split), split, axis)
    # if split is None:
    #     return np.split(input, split, axis=axis) 
    return input


def Tanh(X):
    return np.tanh(X)	


def Transpose(data, perm):
    if not perm:
        return np.transpose(data)
    return np.transpose(data, perm)



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
    'BatchNormalization': BatchNormalization,
    'BitShift': None,
    'Cast': Cast,
    'Ceil': None,
    'Celuk': None,
    'Clip': None,
    'Compress': None,
    'Concat': Concat,
    'ConcatFromSequence': None,
    'Constant': None,
    'ConstantOfShape': None,
    'Conv': Conv,
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
    'Exp': Exp,
    'Expand': None,
    'EyeLike': None,
    'Flatten': None,
    'Floor': None,
    'GRU': None,
    'Gather': Gather,
    'GatherElements': None,
    'GatherND': None,
    'Gemm': None,
    'GlobalAveragePool': GlobalAveragePool,
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
    'LeakyRelu': LeakyRelu,
    'Less': None,
    'Log': Log,
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
    'Mul': Mul,
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
    'Resize': Resize,
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
    'Shape': Shape,
    'Shrink': None,
    'Sigmoid': Sigmoid,
    'Sign': None,
    'Sin': None,
    'Sinh': None,
    'Size': None,
    'Slice': Slice,
    'Softplus': None,
    'Softsign': None,
    'SpaceToDepth': None,
    'Split': Split,
    'SplitToSequence': None,
    'Sqrt': None,
    'Squeeze': None,
    'StringNormalizer': None,
    'Sub': None,
    'Sum': None,
    'Tan': None,
    'Tanh': Tanh ,
    'TfIdfVectorizer': None,
    'ThresholdedRelu': None,
    'Tile': None,
    'TopK': None,
    'Transpose': Transpose,
    'Trilu': None,
    'Unique': None,
    'Unsqueeze': None,
    'Upsample (deprecated)': None,
    'Where': None,
    'Xor': None,
    'Function': None,
    'Celu': None,
    'DynamicQuantizkeLinear': None,
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

# [ minimum input count, maximum input count ]
# -1 means unlimited
argcount = {
    'Abs': [1, 1],
    'Acos': [1, 1],
    'Acosh': [1, 1],
    'Add': [2, 2],
    'And': [2, 2],
    'ArgMax': [1, 1],
    'ArgMin': [1, 1],
    'Asin': [1, 1],
    'Asinh': [1, 1],
    'Atan': [1, 1],
    'Atanh': [1, 1],
    'AveragePool': [1, 1],
    'BatchNormalization': [5, 5],
    'BitShift': [2, 2],
    'Cast': [1, 1],
    'Ceil': [1, 1],
    'Celu': [1, 1],
    'Clip': [1, 3],
    'Compress': [2, 2],
    'Concat': [1, -1],
    'ConcatFromSequence': None,
    'Constant': None,
    'ConstantOfShape': None,
    'Conv': [2, 3],
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
    'Exp': [1, 1],
    'Expand': None,
    'EyeLike': None,
    'Flatten': None,
    'Floor': None,
    'GRU': None,
    'Gather': [2, 2],
    'GatherElements': None,
    'GatherND': None,
    'Gemm': None,
    'GlobalAveragePool': [1, 1],
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
    'LeakyRelu': [1, 1],
    'Less': None,
    'Log': [1, 1],
    'Loop': None,
    'LpNormalization': None,
    'LpPool': None,
    'MatMul': [2, 2],
    'MatMulInteger': None,
    'Max': None,
    'MaxPool': [1, 1],
    'MaxRoiPool': None,
    'MaxUnpool': None,
    'Mean': None,
    'Min': None,
    'Mod': None,
    'Mul': [2, 2],
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
    'Relu': [1, 1],
    'Reshape': [2, 2],
    'Resize': [1, 4],
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
    'Shape': [1, 1],
    'Shrink': None,
    'Sigmoid': [1, 1],
    'Sign': None,
    'Sin': None,
    'Sinh': None,
    'Size': None,
    'Slice': [3, 5],
    'Softplus': None,
    'Softsign': None,
    'SpaceToDepth': None,
    'Split': [1, 2],
    'SplitToSequence': None,
    'Sqrt': None,
    'Squeeze': None,
    'StringNormalizer': None,
    'Sub': None,
    'Sum': None,
    'Tan': None,
    'Tanh': [1, 1],
    'TfIdfVectorizer': None,
    'ThresholdedRelu': None,
    'Tile': None,
    'TopK': None,
    'Transpose': [1, 1],
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
    'Gather': [ _int('axis', 0) ],
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
    'LeakyRelu': [ _float('alpha', 0.01) ],
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
    'Resize': [ _string('coordinate_transformation_mode', 'half_pixel'),
                _float('cubic_coeff_a', -0.75),
                _int('exclude_outside', 0),
                _float('extrapolation_value', 0.0),
                _string('mode', 'nearest'),
                _string('nearest_mode', 'round_prefer_floor'),
               ],
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
    'Split': [ _int('axis', 0) ],
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
    'Transpose': [ _ints('perm', []) ],
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
