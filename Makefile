.PHONY: all test convert mnist mobilenet yolo-tiny

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

convert: # Convert onnx test case to connx
	# Asin
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_asin_example  ../connx/test/data/node/test_asin_example 
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_asin          ../connx/test/data/node/test_asin         
	# Add
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_size1_singleton_broadcast ../connx/test/data/pytorch-operator/test_operator_add_size1_singleton_broadcast 
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_size1_right_broadcast     ../connx/test/data/pytorch-operator/test_operator_add_size1_right_broadcast     
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_broadcast                 ../connx/test/data/pytorch-operator/test_operator_add_broadcast                 
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_size1_broadcast           ../connx/test/data/pytorch-operator/test_operator_add_size1_broadcast           
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_add_uint8                                          ../connx/test/data/node/test_add_uint8                                          
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_add_bcast                                          ../connx/test/data/node/test_add_bcast                                          
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_add                                                ../connx/test/data/node/test_add                                                
	#Sub
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub         ../connx/test/data/node/test_sub        
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub_example ../connx/test/data/node/test_sub_example
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub_bcast   ../connx/test/data/node/test_sub_bcast  
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub_uint8   ../connx/test/data/node/test_sub_uint8  
