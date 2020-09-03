import onnx_connx.cli as cli

def test_mnist():
    cli.main('examples/mnist/*', '-o', 'out/mnist')

def test_mobilenet():
    cli.main('examples/mobilenet/*', '-o', 'out/mobilenet')

def test_yolo_tiny():
    cli.main('examples/yolo-tiny/*', '-o', 'out/yolo-tiny')
