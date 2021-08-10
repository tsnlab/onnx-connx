# ONNX to CONNX Converter
onnx-connx is a tool which converts ONNX to CONNX model. 
And onnx-connx is also a NumPy ONNX Runtime implementation for CONNX.
We don't recommend to use onnx-connx as an ONNX Runtime for NumPy is very (very) slow. 

# For users
## Install onnx-connx via pip
```sh
pip install git+https://github.com/semihlab/onnx-connx
```

## Convert ONNX to CONNX model
python -m onnx\_connx [onnx model]

# For developers
## Prepare development environments
 * python3
 * protobuf-compiler  # to run bin/dump utility
 * onnx               # python package
 * pytest             # python package
 * tabulate           # python package

```sh
$ sudo apt install python3 python3-pip python3-venv
$ python3 -m pip install --user virtualenv
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install onnx
```

## To debug
```sh
onnx-connx$ bin/dump [onnx path]  # This utility will dump onnx or pb to text using protoc
```

## Test
## Run test 
```sh
onnx-connx$ make test
```
## Convert all test cases that connx supports
```sh
onnx-connx$ make convert-test [ONNX_HOME=[onnx install dir]] [CONNX_HOME=[connx source dir]] # ONNX_HOME and CONNX_HOME can be omitted
```

## Convert individual test case
```sh
onnx-connx$ bin/convert [onnx test case path] [connx test case path]
```

## Run connx backend
connx backend will compile the ONNX to CONNX and run it using NumPy operators.

```sh
python -m onnx_connx.backend [onnx model] [[input tensor] ...]
```

## Run MNIST example
```sh
onnx-connx$ cd examples
onnx-connx/examples$ ./download.sh
onnx-connx/examples$ cd ..
onnx-connx$ python -m onnx_connx.backend examples/mnist/model.onnx examples/mnist/input_0.pb
```

# Contribution
See [CONTRIBUTING.md](CONTRIBUTING.md)

# License
CONNX is licensed under GPLv3. See [LICENSE](LICENSE)
