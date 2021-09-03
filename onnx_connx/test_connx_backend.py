import unittest
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
backend_test.exclude(r'test_identity_sequence_cpu')

# Exclude deprecated operators
backend_test.exclude(r'test_scatter_*')
backend_test.exclude(r'test_upsample_*')

# Exclude legacy onnx api
backend_test.exclude(r'test_operator_clip')  # We support above Clip-6

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()
