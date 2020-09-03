.PHONY: all mnist mobilenet yolo-tiny

all: mnist mobilenet yolo-tiny

mnist:
	python -m onnx_connx.cli examples/mnist/* -o out/mnist

mobilenet:
	python -m onnx_connx.cli examples/mobilenet/* -o out/mobilenet

yolo-tiny:
	python -m onnx_connx.cli examples/yolo-tiny/* -o out/yolo-tiny

clean:
	rm -rf out
