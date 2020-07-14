#! /usr/bin/env python

import os
import sys
import ctypes
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda 
import coco_classes as coco_utils
import visualization as visual_utils
import struct
import zmq


DIR_NAME = os.path.dirname(__file__)
ENGINE_FILE = os.path.abspath(os.path.join(
                DIR_NAME, 
                'ssd_mobilenet_v2_coco', 
                'TRT_ssd_mobilenet_v2_coco.bin'
                ))
IMAGE_PATH = ''
COCO_COLORS = coco_utils.COCO_COLORS
COCO_CLASSES_NAME = coco_utils.COCO_CLASSES_NAME
COCO_SELECTION = coco_utils.COCO_SELECTION
OUTPUT_LAYOUT = 7


def preprocess_img(img, shape=(300,300)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    img = img.transpose((2,0,1)).astype(np.float32)
    img = (2.0/255.0) * img - 1.0
    return img


def draw_text(img, text, topleft, color):
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, topleft, font, 0.75, color, 1, cv2.LINE_AA)
    return img


def draw_boxes(img, boxes, conf, clss):
    for bb, cf, cl in zip(boxes, conf, clss):
        cl = int(cl)
        x_min, y_min, x_max, y_max = bb
        color = COCO_COLORS[cl].tolist()
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        txt_loc = (max(x_min+2, 0),max(y_min+15,0))
        cls_name = COCO_CLASSES_NAME[cl]
        txt = '{} {:.2f}'.format(cls_name, cf)
        img = draw_text(img, txt, txt_loc, color)
    return img


def main():
    # Set up zmq
    ctx = zmq.Context()

    # # Subscriber
    # s = ctx.socket(zmq.SUB)
    # s.setsockopt(zmq.SUBSCRIBE, b'')
    # s.setsockopt(zmq.CONFLATE, 1)  # last msg only.
    # s.connect("tcp://localhost:5563")

    # Publisher
    topic = "cam"
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://*:5563")

    # initialize
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    runtime = trt.Runtime(TRT_LOGGER)

    # load engine
    with open(ENGINE_FILE, 'rb') as f:
        buf = f.read()
        engine = runtime.deserialize_cuda_engine(buf)

    # Create buffer
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size,  np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))

        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    context = engine.create_execution_context()

    cap = cv2.VideoCapture(0)
    t_start = time.time()
    fps = 0.0
    while True:
        ret, image = cap.read()
        img_resized = preprocess_img(image)

        # image = cv2.imread(IMAGE_PATH)
        # img_resized = preprocess_img(image)
        np.copyto(host_inputs[0], img_resized.ravel())

        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()

        output = host_outputs[0]
        h, w, c = image.shape
        conf_th = 0.3
        boxes = []
        confs = []
        clss = []

        dat = 0

        for prefix in range(0, len(output), OUTPUT_LAYOUT):
            label = int(output[prefix+1])

            # Check if label is in selection set
            if label not in COCO_SELECTION:
                continue

            # Check confidence threshold 
            conf = output[prefix+2]
            if conf < conf_th:
                continue 

            # Create output byte
            if COCO_CLASSES_NAME[label] == 'person':
                dat = dat | (1 << 1)
            
            if COCO_CLASSES_NAME[label] != 'person':
                dat = dat | (1 << 0)
            
            xmin = int(output[prefix+3] * w)
            ymin = int(output[prefix+4] * h)
            xmax = int(output[prefix+5] * w)
            ymax = int(output[prefix+6] * h)
            boxes.append([xmin, ymin, xmax, ymax])
            confs.append(conf)
            clss.append(label)
        
        pub.send_multipart([
            topic.encode(),
            dat.to_bytes(1, 'little')
        ])
        img = draw_boxes(image, boxes, confs, clss)

        # Get Lidar data
        # data = struct.unpack('<HI', s.recv())
        # img = draw_text(img, "Lidar Dist: %d" % data[0], (h-15, 5), (0,255,0))

        cv2.imshow('TRT_DEMO', img)
        # Calculate FPS
        t_end = time.time()
        curr_fps = 1.0/(t_end - t_start)
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*(0.05))
        t_start = t_end

        # End Program
        key = cv2.waitKey(1)
        if key == 27: # ESC button
            break
    cap.release()
    cv2.destroyAllWindows()

    # clean up cuda
    del stream
    del cuda_outputs
    del cuda_inputs


if __name__ == '__main__':
    main()




    