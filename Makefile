.PHONY: all test mnist mobilenet yolo-tiny clean

NPROC := $(shell nproc)

all: test mnist mobilenet yolo-tiny

test:
	pytest

mnist:
	python -m onnx_connx examples/mnist/* -o out/mnist

mobilenet:
	python -m onnx_connx examples/mobilenet/* -o out/mobilenet

yolo-tiny:
	python -m onnx_connx examples/yolo-tiny/* -o out/yolo-tiny

clean:
	rm -rf out
