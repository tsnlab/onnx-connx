# ONNX to CONNX Converter
onnx-connx is an converting tool which changes ONNX model to CONNX model

# Developers
## Prepare development environments
 * sudo apt install python3 python3-pip python3-venv
 * python3 -m pip install --user virtualenv
 * python3 -m venv venv
 * source venv/bin/activate
 * pip install --upgrade pip
 * pip install onnx

# Compile
python -m onnx_connx [onnx model]

# Utility
 * sudo apt install protobuf-compiler
bin/dump - This utility will dump onnx or pb to text using protoc

# Test
## Dependent libraries
 * pytest
 * tabulate

## Run test 
pytest
> if you want to test in parallel
```shell
pytest --workers `(nproc)` --tests-per-worker auto
```

# Convert example onnxs to connx
make all

# Users
## Install onnx-connx via pip
pip install git+https://github.com/semihlab/onnx-connx

## Run connx backend
connx backend will compile the ONNX to CONNX and run it using Numpy operators.

python -m onnx_connx.backend [onnx model] [[input tensor] ...]

## Run MNIST example
python -m onnx_connx.backend examples/mnist/model.onnx examples/mnist/input_0.pb

# Operator developers
 1. Add the code to opset.py directly if the implementations is few lines.
    Or Make a [Operator].py file in opset directory if the implementation is big.
 2. Import the function in head of opset.py.
 3. Add the function to opset dictionary.
 4. Add [min input count, max input count] array to argcount dictionary.
 5. Add default attributes to attrset dictionary.
 6. Run ONNX test cases(for particular operators).
    pytest -k [test case]
 
    You can find ONNX test cases using find utility as below.
    find ./venv/lib/python3.6/site-packages/onnx/backend/test/ -name 'model.onnx'
    /home/semih/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_add_uint8/model.onnx  
                                                This is the 'test case' name ^^^^^^^^^^^^^^
 7. Run full ONNX test cases. It will take about 30 mins under AMD Ryzen Threadripper 1900X.
    pytest

Some tips
 * We don't care about training parameters. e.g. running_mean, running_var of BatchNormalization.
 * You can get the sample Numpy implementation from Operators.md document in ONNX project.
 * You can get C implementations from CONNX v1.0 repository.

# License
 * onnx-connx is licensed under GPLv3
