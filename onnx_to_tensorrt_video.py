#!/usr/bin/env python2

from __future__ import print_function
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw,Image,ImageFont
import cv2
#from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


TRT_LOGGER = trt.Logger()
text_size = 23
fontText = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",text_size,encoding="utf-8")

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    #print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 22), '{0} {1:.2f}'.format(all_categories[category], score),font = fontText, fill=bbox_color)

    return image_raw

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 1GB
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    ### henry, your onnx and trt file name
    onnx_file_path = './onnx/object.onnx'
    engine_file_path = "./trt/object.trt"
    # Download a dog image and save it to the following file path:
    #input_image_path = download_file('dog.jpg',
    #    'https://github.com/pjreddie/darknet/raw/f86901f6177dfc6116360a13cc06ab680e0c86b0/data/dog.jpg', checksum_reference=None)

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    ### henry, input shape
    input_resolution_yolov3_HW = (416, 416)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3


    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)

    # Load an image from the specified input path, and return it together with  a pre-processed version
    #image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    #shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    ### henry, output shape of your network (1,CHW)
    output_shapes = [(1, 42, 13, 13), (1, 42, 26, 26), (1, 42, 52, 52)]
    
    
    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                        "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                       (59, 119), (116, 90), (156, 198), (373, 326)],
                        "obj_threshold": 0.4,                                               # Threshold for object coverage, float value between 0 and 1
                        "nms_threshold": 0.2,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                        "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        #print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"


        while True:

            prev_time = time.time()
            return_value, frame = vid.read()

            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            start = time.time()
            image_raw, image = preprocessor.process(img_PIL)
            print("***time for pre-process: %.2f s" %(time.time()-start))
            
            shape_orig_WH = image_raw.size
            
            inputs[0].host = image
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            #print(trt_outputs)

            # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]



            # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
            start = time.time()
            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
            #print(classes)
            print("time for post-process: %.2f s***\n" %(time.time()-start))
            # Draw the bounding boxes onto the original input image and save it as a PNG file

            if type(boxes) == type(None):
                obj_detected_img = image_raw
            else:
                obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
            img_OpenCV = cv2.cvtColor(np.asarray(obj_detected_img),cv2.COLOR_RGB2BGR)
            result = np.asarray(img_OpenCV)
            curr_time = time.time()
            exec_time = curr_time - prev_time
            cur_fps = round(1/exec_time,1)
            fps = "FPS: " + str(cur_fps)

            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50, color=(255, 0, 0), thickness=2)
            try:
                count_text = 'finish count:%d'%len(classes)
            except:
                count_text = 'finish count:0'
            cv2.putText(result, text=count_text, org=(15, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0, color=(0, 0, 255), thickness=4)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
if __name__ == '__main__':
    main()
