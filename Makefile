.PHONY: all mnist mobilenet yolo-tiny

all: mnist mobilenet yolo-tiny

mnist:
	python onnx_connx/cli.py examples/mnist/* -o out/mnist

mobilenet:
	python onnx_connx/cli.py examples/mobilenet/* -o out/mobilenet

yolo-tiny:
	python onnx_connx/cli.py examples/yolo-tiny/* -o out/yolo-tiny

clean:
	rm -rf out
