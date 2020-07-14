# Object Detection & Avoidance with 1D Lidar using Nvidia Jetson

## Intro
The goal of this project is to implement object detection using deep learning model provided by TensorFlow and integrate 1D lidar detection info to the output result. The output is sent through CAN or ethernet depending on the avaialble peripherals. 

In order to optimize detection speed, TensorRT (inference optimizer and runtime engine) is implemented to take advantage of the embedded GPU from Jetson boards. [TensorRT Document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#:~:text=Figure%201.,power%20efficiency%2C%20and%20memory%20consumption.) 

## Model
The detection model used in this project is **SSD MobileNet V2 Coco** that can be downloaded from [Tensorflow 1.x Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

Read more on SSD MobileNet [here](https://honingds.com/blog/ssd-single-shot-object-detection-mobilenet-opencv/)

A version of SSD_MobileNet_V2_COCO is provided in **ssd_mobilenet_v2_coco** folder. 

## Set-up Requirement
Refer to [Jetpack-TF Installation Guide](./jetpack_tf_installation.md)

## Build
#### a. Create Inference Engine with Python
Run `object_detection/build_engine.py`, and a `object_detection/TRT_ssd_mobilenet_v2_coco.bin` file will be generated in **object_detection/ssd_mobilenet_v2_coco** folder.
This file is only needed to generated once unless using new frozen model or changes made in build script.

#### b. Deployment
- Run the sample `object_detection/detect_objects.py` script to deploy the tensorrt inference solution.
- This script reads live images from webcam and draw bounding box in detected object to display. The SSD_MobileNet_V2_COCO can detect up to 91 categories, but the interested categories are limited to ['person', 'car', 'bus', 'truck']. See `object_detection/coco_classes.py` for more info. 

## References
- [Tensorrt Document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- https://github.com/opencv/opencv/issues/15074
- https://github.com/AastaNV/TRT_object_detection
- https://github.com/jkjung-avt/tensorrt_demos
- https://jkjung-avt.github.io/jetpack-4.4/
- Local tensorrt sample from Jetpack-4.4: /usr/src/tensorrt/samples/python/uff_ssd
