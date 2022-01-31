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
```sh
python -m onnx\_connx --help                    # to get help message
python -m onnx\_connx [onnx model] [output dir] # to convert onnx to connx
```

# For developers
## Prepare development environments
 * python3
 * onnx               # python package, to run onnx2connx converter
 * protobuf-compiler  # to run bin/dump utility
 * tabulate           # python package, to run test cases
 * pytest-parallel    # python package, to run test cases

```sh
$ sudo apt install python3 python3-pip python3-venv
$ python3 -m pip install --user virtualenv
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install onnx
```
## Debug installation
```sh
pip install git+file:///[path-to-onnx-connx]
```

## Dump onnx to text
```sh
onnx-connx$ bin/dump [onnx path]  # This utility will dump onnx or pb to text using protoc
```

## Test
## Run test 
connx binary must in onnx\_connx, current directory or in PATH environment variable.

```sh
onnx-connx$ make test
```

## Convert individual test case
```sh
onnx-connx$ bin/convert [onnx test case path] [connx test case path]
```

## Run connx backend
connx backend will compile the ONNX to CONNX and run it using connx.
connx binary must in onnx\_connx, current directory or in PATH environment variable.

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
