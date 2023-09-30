import torch
import torchvision
import os
import datetime
import os
import time
import math
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
from PIL import Image
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
from matplotlib.patches import Rectangle
import time
import argparse

from DeepDataMiningLearning.detection import utils
from DeepDataMiningLearning.detection.trainutils import create_aspect_ratio_groups, GroupedBatchSampler

from DeepDataMiningLearning.detection.dataset import get_dataset 
from DeepDataMiningLearning.detection.models import get_torchvision_detection_models, modify_fasterrcnnheader 
from DeepDataMiningLearning.detection.myevaluator import simplemodelevaluate, modelevaluate
from DeepDataMiningLearning.detection.models import CustomRCNN
from DeepDataMiningLearning.detection.plotutils import show_image_bbxyxy, resize, infer_transforms, convert_detections, inference_annotations

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")

# references
# https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/inference.py
# https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/inference_video.py

os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/argo1COCO'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default = 'fasterrcnn_resnet50_fpn_v2', type = str, help = 'e.g. fasterrcnn_resnet50_fpn_v2 or customrcnn_resnet152')
    parser.add_argument('--model_path', default = '/data/cmpe249-fa23/argo1COCO/trainoutput/model_5.pth', type = str,  help = 'path to model')
    parser.add_argument('--testdata_path', default = '/data/cmpe249-fa23/Argoverse/Argoverse-1.1/images/test', type = str,  help = 'path to test dataset')
    parser.add_argument('--num_img', default = 1000, type = int,  help = 'number of images for inferencing')
    parser.add_argument('--video_path', default = '/home/015957045/cmpe249/video_1.mp4', type = str,  help = 'path to video data')
    parser.add_argument('--output_path', default = '/home/015957045/cmpe249/prediction', type = str,  help = 'path to video data')
    return parser

def main(args):
    if args.model_name not in ['fasterrcnn_resnet50_fpn_v2', 'customrcnn_resnet152']:
        raise ValueError("The model is not supported")
    
    # Step 1: Initialize model with the best available weights
    num_classes = 8
    trainable = 0
    CLASS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck', 6: 'traffic_light', 7: 'stop_sign'}
    checkpoint = torch.load(args.model_path)
    if args.model_name == 'fasterrcnn_resnet50_fpn_v2':
        name = args.model_name.split('_')[0]
        model, preprocess, weights, classes = get_torchvision_detection_models(args.model_name)
        model = modify_fasterrcnnheader(model, num_classes, freeze=False)
        model.load_state_dict(checkpoint['model'])
        
    else:
        name = args.model_name.split('_')[0]
        model = CustomRCNN(backbone_modulename="resnet152",
                    trainable_layers=trainable,
                    num_classes=num_classes,
                    out_channels=256,min_size=800,max_size=1333)
        model.load_state_dict(checkpoint['model'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    detection_threshold = 0.9

    # Calculate avg FPS for inferencing test dataset
    image_path_list_test = list(Path(args.testdata_path).glob("*/*/*.jpg"))
    if args.num_img > len(image_path_list_test):
        raise ValueError(f"The number of images for inferencing should be smaller than {len(image_path_list_test)}")
    
    #random.seed(123)
    rand_int = 77 #random.randint(0,len(image_path_list_test)-1)
    print(f"rand_int: {rand_int}")
    frame_count = 0
    total_fps = 0
    image_path_list_test = image_path_list_test[:args.num_img]
    print("Start inferencing test dataset....")
    print(f"The number of test images is {len(image_path_list_test)}")
    for i in range(len(image_path_list_test)):
        temp = image_path_list_test[i]
        orig_image = cv2.imread(str(temp))
        frame_height, frame_width, _ = orig_image.shape
        RESIZE_TO = frame_width
        image_resized = resize(orig_image, RESIZE_TO, square=False)
        image = image_resized.copy()

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image.copy()
        image = infer_transforms(image)

        start_time = time.time()
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image.to(device))
        end_time = time.time()

        # update total_fps
        total_fps += 1/(end_time - start_time)

        # increse frame count
        frame_count += 1

        # Save inference image for an example
        if i == rand_int:
            boxes_xyxy = outputs[0]['boxes'].cpu().detach().numpy()
            ids = outputs[0]['labels'].cpu().detach().numpy()
            scores = outputs[0]['scores'].cpu().detach().numpy()
            title = name + '_inference_example_' + str(rand_int)
            img_save_path = args.output_path + '/' + title + '.jpg'
            show_image_bbxyxy(img, 
                            boxes_xyxy, 
                            ids, 
                            CLASS, 
                            scores,
                            detection_threshold,
                            title, 
                            savefigname = img_save_path)


    print('.........TEST PREDICTIONS FOR TEST IMAGES COMPLETE.........')
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    print('.........................................')


    print('Start inferencing a video')
    video_path = args.video_path
    COLORS = np.random.uniform(0, 255, size=(num_classes, 3))

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    origin_fps = float(cap.get(cv2.CAP_PROP_FPS))

    OUT_DIR = args.output_path
    save_name = name + '_video'
    out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                            cv2.VideoWriter_fourcc(*'mp4v'), origin_fps, 
                            (frame_width, frame_height))

    RESIZE_TO = frame_width
    frame_count = 0
    total_fps = 0
    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            orig_frame = frame.copy()
            frame = resize(frame, RESIZE_TO, square=False)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            # Add batch dimension.
            image = torch.unsqueeze(image, 0)
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                # Get predictions for the current frame.
                outputs = model(image.to(device))
            inf_end_time = time.time()

            # Update total_fps
            forward_pass_time = inf_end_time - start_time
            fps = 1 / forward_pass_time
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            # Load all detection to CPU for further operations.
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            # Carry further only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                draw_boxes, pred_classes, scores, labels = convert_detections(outputs, detection_threshold, CLASS)
                frame = inference_annotations(
                    draw_boxes, 
                    pred_classes, 
                    scores,
                    labels,
                    CLASS, 
                    COLORS, 
                    orig_frame, 
                    frame)
            else:
                frame = orig_frame
                        
            out.write(frame.astype('uint8'))

        else:
            break

    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print('.........TEST PREDICTIONS FOR A VIDEO COMPLETE.........')
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)





    

    
        




