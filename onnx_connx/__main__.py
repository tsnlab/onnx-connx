import argparse
import onnx

from .proto import ConnxModelProto
from .compiler import compile

if __name__ == '__main__':
    compile()

