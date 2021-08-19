#!/bin/sh

curl -O https://media.githubusercontent.com/media/onnx/models/master/vision/classification/mnist/model/mnist-8.tar.gz
curl -O https://media.githubusercontent.com/media/onnx/models/master/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz
curl -O https://media.githubusercontent.com/media/onnx/models/master/vision/object_detection_segmentation/yolov4/model/yolov4.tar.gz

tar fx mnist-8.tar.gz
tar fx mobilenetv2-7.tar.gz
mv mobilenetv2-7 mobilenet
mv mobilenet/mobilenetv2-7.onnx mobilenet/model.onnx
tar fx yolov4.tar.gz
mv yolov4/yolov4.onnx yolov4/model.onnx
