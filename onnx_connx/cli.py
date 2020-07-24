import argparse
import sys
import onnx

def main():
    parser = argparse.ArgumentParser(description='ONNX-CONNX Command Line Interface')
    parser.add_argument('-i', metavar='onnx', type=str, help='an input ONNX model')
    parser.add_argument('-o', metavar='connx', type=str, help='an output CONNX model')

    args = parser.parse_args()

    print('onnx2connx', args)

if __name__ == '__main__':
    main()
