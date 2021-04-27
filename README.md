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

# Convert example onnxs to connx
make all

# Users
## Install onnx-connx via pip
pip install git+https://github.com/semihlab/onnx-connx

# License
 * onnx-connx is licensed under GPLv3
