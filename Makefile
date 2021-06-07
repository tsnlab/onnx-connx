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
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_asin_example  ../connx/test/data/node/test_asin_example
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_asin          ../connx/test/data/node/test_asin
	# Add
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_size1_singleton_broadcast ../connx/test/data/pytorch-operator/test_operator_add_size1_singleton_broadcast
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_size1_right_broadcast     ../connx/test/data/pytorch-operator/test_operator_add_size1_right_broadcast
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_broadcast                 ../connx/test/data/pytorch-operator/test_operator_add_broadcast
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_add_size1_broadcast           ../connx/test/data/pytorch-operator/test_operator_add_size1_broadcast
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_add_uint8                                          ../connx/test/data/node/test_add_uint8
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_add_bcast                                          ../connx/test/data/node/test_add_bcast
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_add                                                ../connx/test/data/node/test_add
	# Sub
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_sub         ../connx/test/data/node/test_sub
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_sub_example ../connx/test/data/node/test_sub_example
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_sub_bcast   ../connx/test/data/node/test_sub_bcast
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_sub_uint8   ../connx/test/data/node/test_sub_uint8
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
	# Conv
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-operator/test_operator_conv                     ../connx/test/data/pytorch-operator/test_operator_conv
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_dilated                   ../connx/test/data/pytorch-converted/test_Conv2d_dilated
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d                           ../connx/test/data/pytorch-converted/test_Conv2d
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_groups_thnn               ../connx/test/data/pytorch-converted/test_Conv2d_groups_thnn
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d                           ../connx/test/data/pytorch-converted/test_Conv1d
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_depthwise                 ../connx/test/data/pytorch-converted/test_Conv2d_depthwise
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d_pad1size1                 ../connx/test/data/pytorch-converted/test_Conv1d_pad1size1
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_depthwise_strided         ../connx/test/data/pytorch-converted/test_Conv2d_depthwise_strided
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d_pad2                      ../connx/test/data/pytorch-converted/test_Conv1d_pad2
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv3d_dilated_strided           ../connx/test/data/pytorch-converted/test_Conv3d_dilated_strided
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_depthwise_with_multiplier ../connx/test/data/pytorch-converted/test_Conv2d_depthwise_with_multiplier
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv3d                           ../connx/test/data/pytorch-converted/test_Conv3d
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv3d_no_bias                   ../connx/test/data/pytorch-converted/test_Conv3d_no_bias
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_strided                   ../connx/test/data/pytorch-converted/test_Conv2d_strided
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv3d_stride                    ../connx/test/data/pytorch-converted/test_Conv3d_stride
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_padding                   ../connx/test/data/pytorch-converted/test_Conv2d_padding
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_no_bias                   ../connx/test/data/pytorch-converted/test_Conv2d_no_bias
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_groups                    ../connx/test/data/pytorch-converted/test_Conv2d_groups
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv3d_stride_padding            ../connx/test/data/pytorch-converted/test_Conv3d_stride_padding
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv3d_dilated                   ../connx/test/data/pytorch-converted/test_Conv3d_dilated
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d_stride                    ../connx/test/data/pytorch-converted/test_Conv1d_stride
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_ConvTranspose2d                  ../connx/test/data/pytorch-converted/test_ConvTranspose2d
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv3d_groups                    ../connx/test/data/pytorch-converted/test_Conv3d_groups
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d_pad1                      ../connx/test/data/pytorch-converted/test_Conv1d_pad1
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d_pad2size1                 ../connx/test/data/pytorch-converted/test_Conv1d_pad2size1
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d_groups                    ../connx/test/data/pytorch-converted/test_Conv1d_groups
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv2d_depthwise_padded          ../connx/test/data/pytorch-converted/test_Conv2d_depthwise_padded
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_Conv1d_dilated                   ../connx/test/data/pytorch-converted/test_Conv1d_dilated
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/pytorch-converted/test_ConvTranspose2d_no_bias          ../connx/test/data/pytorch-converted/test_ConvTranspose2d_no_bias
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_basic_conv_with_padding                       ../connx/test/data/node/test_basic_conv_with_padding
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_conv_with_strides_padding                     ../connx/test/data/node/test_conv_with_strides_padding
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_conv_with_strides_no_padding                  ../connx/test/data/node/test_conv_with_strides_no_padding
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_conv_with_autopad_same                        ../connx/test/data/node/test_conv_with_autopad_same
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_basic_conv_without_padding                    ../connx/test/data/node/test_basic_conv_without_padding
	bin/convert ~/venv/lib/python3.8/site-packages/onnx/backend/test/data/node/test_conv_with_strides_and_asymmetric_padding      ../connx/test/data/node/test_conv_with_strides_and_asymmetric_padding
