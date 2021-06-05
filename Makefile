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
	# Sub
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub         ../connx/test/data/node/test_sub        
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub_example ../connx/test/data/node/test_sub_example
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub_bcast   ../connx/test/data/node/test_sub_bcast  
	bin/convert ~/venv/lib/python3.6/site-packages/onnx/backend/test/data/node/test_sub_uint8   ../connx/test/data/node/test_sub_uint8  
	# MaxPool
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_maxpool                   ../connx/test/data/pytorch-operator/test_operator_maxpool
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool2d                         ../connx/test/data/pytorch-converted/test_MaxPool2d
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool1d_stride_padding_dilation ../connx/test/data/pytorch-converted/test_MaxPool1d_stride_padding_dilation
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool3d_stride_padding          ../connx/test/data/pytorch-converted/test_MaxPool3d_stride_padding
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool1d                         ../connx/test/data/pytorch-converted/test_MaxPool1d
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool3d                         ../connx/test/data/pytorch-converted/test_MaxPool3d
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool3d_stride                  ../connx/test/data/pytorch-converted/test_MaxPool3d_stride
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool1d_stride                  ../connx/test/data/pytorch-converted/test_MaxPool1d_stride
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_MaxPool2d_stride_padding_dilation ../connx/test/data/pytorch-converted/test_MaxPool2d_stride_padding_dilation
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_precomputed_same_upper              ../connx/test/data/node/test_maxpool_2d_precomputed_same_upper
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_precomputed_pads                    ../connx/test/data/node/test_maxpool_2d_precomputed_pads
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_uint8                               ../connx/test/data/node/test_maxpool_2d_uint8
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_1d_default                             ../connx/test/data/node/test_maxpool_1d_default
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_same_upper                          ../connx/test/data/node/test_maxpool_2d_same_upper
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_with_argmax_2d_precomputed_pads        ../connx/test/data/node/test_maxpool_with_argmax_2d_precomputed_pads
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_dilations                           ../connx/test/data/node/test_maxpool_2d_dilations
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_same_lower                          ../connx/test/data/node/test_maxpool_2d_same_lower
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_pads                                ../connx/test/data/node/test_maxpool_2d_pads
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_precomputed_strides                 ../connx/test/data/node/test_maxpool_2d_precomputed_strides
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_ceil                                ../connx/test/data/node/test_maxpool_2d_ceil
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_default                             ../connx/test/data/node/test_maxpool_2d_default
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_2d_strides                             ../connx/test/data/node/test_maxpool_2d_strides
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_with_argmax_2d_precomputed_strides     ../connx/test/data/node/test_maxpool_with_argmax_2d_precomputed_strides
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_maxpool_3d_default                             ../connx/test/data/node/test_maxpool_3d_default
