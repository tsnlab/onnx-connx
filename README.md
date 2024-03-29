# ONNX to CONNX Converter

onnx-connx is a tool

 * ONNX model to CONNX model compiler
 * ONNX backend using CONNX engine


## For users

### Install onnx-connx via pip

```sh
pip install git+https://github.com/tsnlab/onnx-connx
```

### Convert ONNX to CONNX model

```sh
python -m onnx\_connx --help                    # to get help message
python -m onnx\_connx [onnx model] [output dir] # to convert onnx to connx
```

## For developers

### Prepare development environments
 * python3
 * protobuf-compiler  # to run bin/dump utility
 * poetry

```sh
$ sudo apt install python3 python3-pip
$ poetry install
```

### Dump onnx to text
2 ways to dump onnx or pb file to text.

```sh
onnx-connx$ bin/dump [onnx path]  # This utility will dump onnx or pb to text using protoc
onnx-connx$ onnx2connx -d [onnx path]  # This utility will dump onnx or pb to text using onnx_connx
```

### Test

#### Run ONNX test cases in your environment

```sh
onnx-connx$ cp [connx binary path] onnx_connx/connx
onnx-connx$ pytest
```

#### Run ONNX test cases for specific onnx version

```sh
onnx-connx$ cp [connx binary path] onnx_connx/connx
onnx-connx$ bin/test [onnx version]

```

#### Tested ONNX versions

|ONNX Version|Passed test cases|
|------------|-----------------|
|   1.5.0    |       217       |
|   1.6.0    |       243       |
|   1.7.0    |       259       |
|   1.8.1    |       260       |
|   1.9.0    |       268       |
|   1.10.2   |       278       |
|   1.11.0   |       278       |

### Benchmark operator

```sh
# First, Install onnx-connx
pip instal -e .

# Run benchmark for specific operator test case
bin/benchmark-operator xor

# Results are stored in "results" directory (by default)
```

## Run connx backend

### Install connx manually

If you want to use specific version of [connx][], You need to install it separately.

[connx]: https://github.com/tsnlab/connx

```sh
onnx-connx$ git clone https://github.com/tsnlab/connx && cd connx
onnx-connx$ pip install .
```
Or you can install connx from git directly using pip.

```sh
$ pip install git+https://github.com/tsnlab/connx
```

### Run backend

connx backend will compile the ONNX to CONNX and run it using connx.
connx binary must in `onnx_connx`, current directory or in PATH environment variable.

```sh
connx-run [onnx model] [[input tensor] ...]
```

### Run MNIST example

```sh
onnx-connx$ cd examples
onnx-connx/examples$ ./download.sh
onnx-connx/examples$ cd ..
onnx-connx$ connx-run examples/mnist/model.onnx examples/mnist/input_0.pb
```

## Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

CONNX is licensed under GPLv3. See [LICENSE](LICENSE)
