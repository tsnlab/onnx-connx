.PHONY: all mnist mobilenet

all: mnist mobilenet

mnist:
	python onnx_connx/cli.py examples/mnist/* -o out/mnist

mobilenet:
	python onnx_connx/cli.py examples/mnist/* -o out/mobilenet

clean:
	rm -f out
