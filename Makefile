.PHONY: all test convert mnist mobilenet yolo-tiny convert-examples convert-test

all: test mnist mobilenet yolo-tiny

NPROC := $(shell nproc)

test:
	pytest --workers $(NPROC) --tests-per-worker auto

mnist:
	python -m onnx_connx examples/mnist/* -o out/mnist

mobilenet:
	python -m onnx_connx examples/mobilenet/* -o out/mobilenet

yolo-tiny:
	python -m onnx_connx examples/yolo-tiny/* -o out/yolo-tiny

clean:
	rm -rf out

ONNX_HOME ?= $(shell python3 -c 'import onnx; print(onnx.__path__[0])')
CONNX_HOME ?= $(CONNX_HOME)/

convert-examples: # Convert examples to connx
	bin/convert examples/mnist     $(CONNX_HOME)/examples/mnist
	bin/convert examples/mobilenet $(CONNX_HOME)/examples/mobilenet
	bin/convert examples/yolov4    $(CONNX_HOME)/examples/yolov4

convert-test: # Convert onnx test case to connx
	# Asin
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_asin_example  $(CONNX_HOME)/test/data/node/test_asin_example
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_asin          $(CONNX_HOME)/test/data/node/test_asin
	# Add
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-operator/test_operator_add_size1_singleton_broadcast $(CONNX_HOME)/test/data/pytorch-operator/test_operator_add_size1_singleton_broadcast
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-operator/test_operator_add_size1_right_broadcast     $(CONNX_HOME)/test/data/pytorch-operator/test_operator_add_size1_right_broadcast
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-operator/test_operator_add_broadcast                 $(CONNX_HOME)/test/data/pytorch-operator/test_operator_add_broadcast
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-operator/test_operator_add_size1_broadcast           $(CONNX_HOME)/test/data/pytorch-operator/test_operator_add_size1_broadcast
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_add_uint8                                          $(CONNX_HOME)/test/data/node/test_add_uint8
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_add_bcast                                          $(CONNX_HOME)/test/data/node/test_add_bcast
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_add                                                $(CONNX_HOME)/test/data/node/test_add
	# Sub
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_sub         $(CONNX_HOME)/test/data/node/test_sub
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_sub_example $(CONNX_HOME)/test/data/node/test_sub_example
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_sub_bcast   $(CONNX_HOME)/test/data/node/test_sub_bcast
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_sub_uint8   $(CONNX_HOME)/test/data/node/test_sub_uint8
	# MaxPool
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-operator/test_operator_maxpool                   $(CONNX_HOME)/test/data/pytorch-operator/test_operator_maxpool
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool2d                         $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool2d
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool1d_stride_padding_dilation $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool1d_stride_padding_dilation
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool3d_stride_padding          $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool3d_stride_padding
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool1d                         $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool1d
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool3d                         $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool3d
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool3d_stride                  $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool3d_stride
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool1d_stride                  $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool1d_stride
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_MaxPool2d_stride_padding_dilation $(CONNX_HOME)/test/data/pytorch-converted/test_MaxPool2d_stride_padding_dilation
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_precomputed_same_upper              $(CONNX_HOME)/test/data/node/test_maxpool_2d_precomputed_same_upper
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_precomputed_pads                    $(CONNX_HOME)/test/data/node/test_maxpool_2d_precomputed_pads
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_uint8                               $(CONNX_HOME)/test/data/node/test_maxpool_2d_uint8
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_1d_default                             $(CONNX_HOME)/test/data/node/test_maxpool_1d_default
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_same_upper                          $(CONNX_HOME)/test/data/node/test_maxpool_2d_same_upper
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_with_argmax_2d_precomputed_pads        $(CONNX_HOME)/test/data/node/test_maxpool_with_argmax_2d_precomputed_pads
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_dilations                           $(CONNX_HOME)/test/data/node/test_maxpool_2d_dilations
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_same_lower                          $(CONNX_HOME)/test/data/node/test_maxpool_2d_same_lower
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_pads                                $(CONNX_HOME)/test/data/node/test_maxpool_2d_pads
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_precomputed_strides                 $(CONNX_HOME)/test/data/node/test_maxpool_2d_precomputed_strides
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_ceil                                $(CONNX_HOME)/test/data/node/test_maxpool_2d_ceil
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_default                             $(CONNX_HOME)/test/data/node/test_maxpool_2d_default
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_2d_strides                             $(CONNX_HOME)/test/data/node/test_maxpool_2d_strides
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_with_argmax_2d_precomputed_strides     $(CONNX_HOME)/test/data/node/test_maxpool_with_argmax_2d_precomputed_strides
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_maxpool_3d_default                             $(CONNX_HOME)/test/data/node/test_maxpool_3d_default
	# Conv
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-operator/test_operator_conv                     $(CONNX_HOME)/test/data/pytorch-operator/test_operator_conv
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_dilated                   $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_dilated
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d                           $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_groups_thnn               $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_groups_thnn
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d                           $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_depthwise                 $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_depthwise
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d_pad1size1                 $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d_pad1size1
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_depthwise_strided         $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_depthwise_strided
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d_pad2                      $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d_pad2
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv3d_dilated_strided           $(CONNX_HOME)/test/data/pytorch-converted/test_Conv3d_dilated_strided
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_depthwise_with_multiplier $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_depthwise_with_multiplier
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv3d                           $(CONNX_HOME)/test/data/pytorch-converted/test_Conv3d
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv3d_no_bias                   $(CONNX_HOME)/test/data/pytorch-converted/test_Conv3d_no_bias
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_strided                   $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_strided
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv3d_stride                    $(CONNX_HOME)/test/data/pytorch-converted/test_Conv3d_stride
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_padding                   $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_padding
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_no_bias                   $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_no_bias
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_groups                    $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_groups
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv3d_stride_padding            $(CONNX_HOME)/test/data/pytorch-converted/test_Conv3d_stride_padding
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv3d_dilated                   $(CONNX_HOME)/test/data/pytorch-converted/test_Conv3d_dilated
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d_stride                    $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d_stride
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv3d_groups                    $(CONNX_HOME)/test/data/pytorch-converted/test_Conv3d_groups
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d_pad1                      $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d_pad1
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d_pad2size1                 $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d_pad2size1
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d_groups                    $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d_groups
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv2d_depthwise_padded          $(CONNX_HOME)/test/data/pytorch-converted/test_Conv2d_depthwise_padded
	bin/convert $(ONNX_HOME)/backend/test/data/pytorch-converted/test_Conv1d_dilated                   $(CONNX_HOME)/test/data/pytorch-converted/test_Conv1d_dilated
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_basic_conv_with_padding                       $(CONNX_HOME)/test/data/node/test_basic_conv_with_padding
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_conv_with_strides_padding                     $(CONNX_HOME)/test/data/node/test_conv_with_strides_padding
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_conv_with_strides_no_padding                  $(CONNX_HOME)/test/data/node/test_conv_with_strides_no_padding
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_conv_with_autopad_same                        $(CONNX_HOME)/test/data/node/test_conv_with_autopad_same
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_basic_conv_without_padding                    $(CONNX_HOME)/test/data/node/test_basic_conv_without_padding
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_conv_with_strides_and_asymmetric_padding      $(CONNX_HOME)/test/data/node/test_conv_with_strides_and_asymmetric_padding
	
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_identity                                      $(CONNX_HOME)/test/data/node/test_identity
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_gather_0 $(CONNX_HOME)/test/data/node/test_gather_0
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_nonzero_example $(CONNX_HOME)/test/data/node/test_nonzero_example
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_greater_equal $(CONNX_HOME)/test/data/node/test_greater_equal
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_greater_equal_bcast $(CONNX_HOME)/test/data/node/test_greater_equal_bcast
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_greater_equal_expanded $(CONNX_HOME)/test/data/node/test_greater_equal_expanded
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_greater_equal_bcast_expanded $(CONNX_HOME)/test/data/node/test_greater_equal_bcast_expanded
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_equal $(CONNX_HOME)/test/data/node/test_equal
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_equal_bcast $(CONNX_HOME)/test/data/node/test_equal_bcast
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or2d $(CONNX_HOME)/test/data/node/test_or2d
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or3d $(CONNX_HOME)/test/data/node/test_or3d
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or4d $(CONNX_HOME)/test/data/node/test_or4d
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or_bcast3v1d $(CONNX_HOME)/test/data/node/test_or_bcast3v1d
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or_bcast3v2d $(CONNX_HOME)/test/data/node/test_or_bcast3v2d
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or_bcast4v2d $(CONNX_HOME)/test/data/node/test_or_bcast4v2d
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or_bcast4v3d $(CONNX_HOME)/test/data/node/test_or_bcast4v3d
	bin/convert  $(ONNX_HOME)/backend/test/data/node/test_or_bcast4v4d $(CONNX_HOME)/test/data/node/test_or_bcast4v4d
	# seq is not supported yest
	#bin/convert $(ONNX_HOME)/backend/test/data/node/test_identity_sequence                             $(CONNX_HOME)/test/data/node/test_identity_sequence
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_scales_cubic/ $(CONNX_HOME)/test/data/node/test_resize_downsample_scales_cubic/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_scales_cubic_align_corners/ $(CONNX_HOME)/test/data/node/test_resize_downsample_scales_cubic_align_corners/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_scales_cubic_A_n0p5_exclude_outside/ $(CONNX_HOME)/test/data/node/test_resize_downsample_scales_cubic_A_n0p5_exclude_outside/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_scales_linear/ $(CONNX_HOME)/test/data/node/test_resize_downsample_scales_linear/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_scales_linear_align_corners/ $(CONNX_HOME)/test/data/node/test_resize_downsample_scales_linear_align_corners/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_scales_nearest/ $(CONNX_HOME)/test/data/node/test_resize_downsample_scales_nearest/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_sizes_cubic/ $(CONNX_HOME)/test/data/node/test_resize_downsample_sizes_cubic/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_sizes_linear_pytorch_half_pixel/ $(CONNX_HOME)/test/data/node/test_resize_downsample_sizes_linear_pytorch_half_pixel/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_sizes_nearest/ $(CONNX_HOME)/test/data/node/test_resize_downsample_sizes_nearest/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn/ $(CONNX_HOME)/test/data/node/test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn/
	# bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_tf_crop_and_resize/ $(CONNX_HOME)/test/data/node/test_resize_tf_crop_and_resize/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_scales_cubic/ $(CONNX_HOME)/test/data/node/test_resize_upsample_scales_cubic/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_scales_cubic_align_corners/ $(CONNX_HOME)/test/data/node/test_resize_upsample_scales_cubic_align_corners/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_scales_cubic_A_n0p5_exclude_outside/ $(CONNX_HOME)/test/data/node/test_resize_upsample_scales_cubic_A_n0p5_exclude_outside/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_scales_cubic_asymmetric/ $(CONNX_HOME)/test/data/node/test_resize_upsample_scales_cubic_asymmetric/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_scales_linear/ $(CONNX_HOME)/test/data/node/test_resize_upsample_scales_linear/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_scales_linear_align_corners/ $(CONNX_HOME)/test/data/node/test_resize_upsample_scales_linear_align_corners/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_scales_nearest/ $(CONNX_HOME)/test/data/node/test_resize_upsample_scales_nearest/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_sizes_cubic/ $(CONNX_HOME)/test/data/node/test_resize_upsample_sizes_cubic/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_sizes_nearest/ $(CONNX_HOME)/test/data/node/test_resize_upsample_sizes_nearest/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_sizes_nearest_ceil_half_pixel/ $(CONNX_HOME)/test/data/node/test_resize_upsample_sizes_nearest_ceil_half_pixel/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_sizes_nearest_floor_align_corners/ $(CONNX_HOME)/test/data/node/test_resize_upsample_sizes_nearest_floor_align_corners/
	bin/convert $(ONNX_HOME)/backend/test/data/node/test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric/ $(CONNX_HOME)/test/data/node/test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric/

