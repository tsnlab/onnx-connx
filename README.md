# ONNX to CONNX Converter
onnx-connx is a tool

 * ONNX model to CONNX model compiler
 * ONNX backend using CONNX engine

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
 * protobuf-compiler  # to run bin/dump utility

```sh
$ sudo apt install python3 python3-pip
$ python3 -m pip install --user virtualenv
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip pytest tabulate onnx
```

## Debug installation
```sh
pip install git+file:///[path-to-onnx-connx]
```

## Dump onnx to text
2 ways to dump onnx or pb file to text.

```sh
onnx-connx$ bin/dump [onnx path]  # This utility will dump onnx or pb to text using protoc
onnx-connx$ python -m onnx_connx -d [onnx path]  # This utility will dump onnx or pb to text using onnx_connx
```

## Test
## Run ONNX test cases in your environment
```sh
onnx-connx$ cp [connx binary path] connx
onnx-connx$ pytest
```

## Run ONNX test cases for specific onnx version
```sh
onnx-connx$ cp [connx binary path] connx
onnx-connx$ bin/test [onnx version]

```

## Tested ONNX versions
|ONNX Version|Passed test cases|
|------------|-----------------|
|   1.7.0    |       100%      |
|   1.8.1    |       100%      |
|   1.9.0    |       100%      |
|   1.10.2   |       100%      |
|   1.11.0   |       100%      |

# Run connx backend
# Run backend
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
