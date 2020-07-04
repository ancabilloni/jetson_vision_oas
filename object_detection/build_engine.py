#! /usr/bin/env python3

"""
This script is to convert the SSD model (pb) to UFF
and builds the TensorRT engine.

Input: .pb
Output: .bin
"""

import os
import ctypes
import uff
import tensorrt as trt
import graphsurgeon as gs


DIR_NAME = os.path.dirname(__file__)
LIB_FILE = os.path.abspath(os.path.join(DIR_NAME,'lib','libflattenconcat.so'))
MODEL_SPECS = {
    'ssd_mobilenet_v2_coco': {
        'input_pb':   os.path.abspath(os.path.join(DIR_NAME, 'ssd_mobilenet_v2_coco', 'frozen_inference_graph.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(DIR_NAME, 'ssd_mobilenet_v2_coco', 'tmp.uff')),
        'output_bin': os.path.abspath(os.path.join(DIR_NAME, 'ssd_mobilenet_v2_coco', 'TRT_ssd_mobilenet_v2_coco.bin')),
        'num_classes': 91,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [1,0,2]
    }
}

INPUT_DIMS = (3, 300, 300)
DEBUG_UFF = False

def add_plugin(graph, spec):
    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape= (1,) + INPUT_DIMS
    )

    PriorBox = gs.create_plugin_node(
        name="MultipleGridAnchorGenerator",
        op="GridAnchor_TRT",
        minSize=spec['min_size'],
        maxSize=spec['max_size'],
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1, 0.1, 0.2, 0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=spec['num_classes'],
        inputOrder=spec['input_order'],
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT"
    )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT"
    )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        "Concatenate": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    graph.collapse_namespaces(namespace_plugin_map)
    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    
    return graph


def main():
    # initialize
    ctypes.CDLL(LIB_FILE)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # compile the model into TensorRT engine
    model = 'ssd_mobilenet_v2_coco'
    spec = MODEL_SPECS[model]
    if not os.path.exists(spec['tmp_uff']):
        dynamic_graph = add_plugin(gs.DynamicGraph(spec['input_pb']), spec)
        uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), output_nodes=['NMS'], output_filename=spec['tmp_uff'], text=True, debug_mode=DEBUG_UFF)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', INPUT_DIMS)
        parser.register_output('MarkOutput_0')
        parser.parse(spec['tmp_uff'], network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(spec['output_bin'], 'wb') as f:
            f.write(buf)


if __name__ == '__main__':
    main()