import unittest

import onnx
import onnx.backend.test

from .backend import Backend


pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.runner.Runner(Backend, __name__)

# Exclude training operators
backend_test.exclude(r'test_adagrad_*')
backend_test.exclude(r'test_adam_*')
backend_test.exclude(r'test_gradient_*')
backend_test.exclude(r'test_momentum_*')
backend_test.exclude(r'nesterov_momentum_*')
backend_test.exclude(r'test_batchnorm_epsilon_training_mode')
backend_test.exclude(r'test_batchnorm_example_training_mode')

# Exclude not supported operators
backend_test.exclude(r'test_identity_sequence_cpu')
backend_test.exclude(r'test_cast.*FLOAT16.*')
backend_test.exclude(r'test_cast.*BFLOAT16.*')
backend_test.exclude(r'test_cast.*STRING.*')
backend_test.exclude(r'test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn')

# Exclude not supported types
if onnx.__version__ == '1.11.0':
    backend_test.exclude(r'identity_opt')
    backend_test.exclude(r'sequence')

# Exclude not supported features Resize TF_CROP_AND_RESIZE
backend_test.exclude(r'test_resize_tf_crop_and_resize')  # We support above Clip-6

# Exclude deprecated operators
backend_test.exclude(r'test_scatter_*')
backend_test.exclude(r'test_upsample_*')

# Exclude malformed test cases
if onnx.__version__ == '1.6.0':
    backend_test.exclude('test_resize_upsample_sizes_nearest_ceil_half_pixel')  # nearest_mode=ceil is missing
    backend_test.exclude('test_resize_upsample_sizes_nearest_floor_align_corners')  # nearest_mode=floor is missing
    backend_test.exclude(
        'test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric')  # nearest_mode=round_prefer_ceil is missing

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()
