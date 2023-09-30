#ref: https://github.com/lkk688/MultiModalDetector/blob/master/Myutils/plotresults.py
import cv2
#from utils.plotresults import show_image_bbxyxy
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle
#%matplotlib inline
from PIL import Image 
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms as transforms
INSTANCE_Color = {
    'Unknown':'black', 'Vehicles':'red', 'Pedestrians':'green', 'Cyclists':'purple'
}#'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'



def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    #color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    color = [int((p * ((label+5*label) ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES):
    numclasses=len(INSTANCE_CATEGORY_NAMES)
    pred_labels=[]
    for pred_i in list(pred_ids):
        if pred_i >= numclasses:
            pred_labels.append('Unknown')
        else:
            pred_labels.append(INSTANCE_CATEGORY_NAMES[pred_i])
    return pred_labels

def matplotlibshow_image_bbxyxy(image, pred_bbox, pred_ids, title, INSTANCE_CATEGORY_NAMES, savefigname=None):
    """Show a camera image (HWC format) and the given camera labels."""
        
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    boxnum=len(pred_bbox)
    #print(boxnum)
    if len(pred_ids)<1:
        print("No object detected")
        return image
    else:
        #pred_labels = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred_ids) ]
        pred_labels = convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES)
        #print(pred_labels)
        for i in range(boxnum):#patch in pred_bbox:
            patch=pred_bbox[i]
            #print(patch)
            colorlabel=compute_color_for_labels(pred_ids[i]) #INSTANCE_Color[label]
            #print(colorlabel)#RGB value 0-255
            colorlabelnormalized = [float(i)/255 for i in colorlabel] #0-1
            label=pred_labels[i]
            #print(label)
            ax.add_patch(Rectangle(
            xy=(patch[0], patch[1]), #xmin ymin
            width=patch[2] - patch[0],
            height=patch[3] - patch[1],
            linewidth=4,
            edgecolor=colorlabelnormalized,#"red",
            facecolor='none'))
            ax.text(patch[0], patch[1], label, color=colorlabelnormalized, fontsize=15)
            #ax.text(patch[0][0], patch[0][1], label, bbox=dict(facecolor='red', alpha=0.5))#fontsize=8)
        
    ax.imshow(image)
    
    ax.title.set_text(title)
    ax.grid(False)
    ax.axis('off')
    
    if savefigname is not None:
        fig.savefig(savefigname)
    
    #fig.savefig(f"output/test_frame_{i}.png", dpi=fig.dpi)
#     plt.show()

from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

def pil_tonp(img_pil, outputformat='CHW'):
    #convert PIL image to numpy
    a = np.asarray(img_pil)
    print("input image shape", a.shape) #HWC format (height, width, color channels)
    if outputformat=='CHW':
        imgdata = a.transpose((2, 0, 1)) #CHW (color channels, height, width)
    elif outputformat=='HWC':
        imgdata = a
    return imgdata

import cv2
def npimage_RGBchange(imgdata, fromformat='BGR', toformat='RGB'): 
    #imgdata is HWC format
    im = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    return im

##torch read_image get format CHW (color channels, height, width)
#pred_bbox_tensor in (xmin, ymin, xmax, ymax) format
def drawbbox_topil(imgdata_np, pred_bbox_np, labels_str, colors='red'):
    #imgdata_np CHW format
    pred_bbox_tensor = torch.from_numpy(pred_bbox_np)
    box = draw_bounding_boxes(torch.from_numpy(imgdata_np), boxes=pred_bbox_tensor,
                            labels=labels_str,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    return im

def draw_boxes(image, pred_bbox, pred_ids, pred_score, INSTANCE_CATEGORY_NAMES):
    boxnum=len(pred_bbox)
    #print(boxnum)
    if len(pred_ids)<1:
        print("No object detected")
        return image
    else:
        #pred_labels = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred_ids) ]
        pred_labels = convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES)
        #print(pred_labels)
        pred_score_str=["%.2f" % i for i in pred_score]
        for i in range(boxnum):#patch in pred_bbox:
            patch=pred_bbox[i] # [ (xmin, ymin), (xmax, ymax)]
            #patch[0] (xmin, ymin)
            #patch[1] (xmax, ymax)
            x1=int(patch[0][0])
            y1=int(patch[0][1])
            x2=int(patch[1][0])
            y2=int(patch[1][1]) #cv2.rectangle need int input not float
            colorlabel=compute_color_for_labels(pred_ids[i]) #RGB value 0-255
            label=pred_labels[i]+" "+pred_score_str[i]
            labelscale=1
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, labelscale , 2)[0] #font scale: 1, font_thickness: 1
            cv2.rectangle(image, (x1, y1), (x2,y2), colorlabel, 2)
            cv2.rectangle(image, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), colorlabel, -1) #-1 is fill the rectangle
            cv2.putText(image,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, labelscale, [255,255,255], 2)
    return image

def draw_trackingboxes(image, pred_bbox, identities=None, track_class=None, INSTANCE_CATEGORY_NAMES=None):
    boxnum=len(pred_bbox)
    #print(boxnum)
    if boxnum<1:
        print("No object detected")
        return image
    else:
        if track_class is not None:
            pred_labels = convertIDtolabel(track_class, INSTANCE_CATEGORY_NAMES)
        for i in range(boxnum):#patch in pred_bbox:
            patch=pred_bbox[i] # [ xmin, ymin, xmax, ymax]
            #patch[0] (xmin, ymin)
            #patch[1] (xmax, ymax)
            x1=int(patch[0])
            y1=int(patch[1])
            x2=int(patch[2])
            y2=int(patch[3]) #cv2.rectangle need int input not float
            id = int(identities[i]) if identities is not None else 0    

            
            if track_class is not None:
                label=pred_labels[i]+" T:"+str(id)
                colorlabel=compute_color_for_labels(track_class[i]) #RGB value 0-255
            else:
                label="T:"+str(id)
                colorlabel = compute_color_for_labels(id)
            labelscale=1
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, labelscale , 2)[0] #font scale: 1, font_thickness: 1
            cv2.rectangle(image, (x1, y1), (x2,y2), colorlabel, 2)
            cv2.rectangle(image, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), colorlabel, -1) #-1 is fill the rectangle
            cv2.putText(image,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, labelscale, [255,255,255], 2)
    return image



def show_imagewithscore_bbxyxy(image, pred_bbox, pred_ids, pred_score, title, INSTANCE_CATEGORY_NAMES, savefigname=None):
    """Show a camera image and the given camera labels."""
        
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    boxnum=len(pred_bbox)
    #print(boxnum)
    #pred_ids may contain 80, but INSTANCE_CATEGORY_NAMES only has 79
    #pred_labels = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred_ids)]
    pred_labels = convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES)

    pred_score_str=["%.2f" % i for i in pred_score]
    #print(pred_labels)
    for i in range(boxnum):#patch in pred_bbox:
        patch=pred_bbox[i]
        #print(patch)
        colorlabel=compute_color_for_labels(pred_ids[i]) #INSTANCE_Color[label]
        #print(colorlabel)#RGB value 0-255
        colorlabelnormalized = [float(i)/255 for i in colorlabel] #0-1
        label=pred_labels[i]+" "+pred_score_str[i]
        #print(label)
        ax.add_patch(Rectangle(
        xy=patch[0],#(patch[0], patch[1]), #xmin ymin
        width=patch[1][0]-patch[0][0],#patch[2] - patch[0],
        height=patch[1][1]-patch[0][1],#patch[3] - patch[1],
        linewidth=3,
        edgecolor=colorlabelnormalized,#"red",
        facecolor='none'))
        #ax.text(patch[0][0], patch[0][1], label, color=colorlabelnormalized, fontsize=14)
        ax.text(patch[0][0], patch[0][1], label, bbox=dict(facecolor=colorlabelnormalized, alpha=0.4), fontsize=14)#fontsize=8)
        
        
    ax.imshow(image)
    
    ax.title.set_text(title)
    ax.grid(False)
    ax.axis('off')
    
    if savefigname is not None:
        fig.savefig(savefigname)
    
def show_image_bbxyxy(image, 
                    pred_bbox, 
                    pred_ids, 
                    class_dic, 
                    prob, 
                    detection_threshold, 
                    title, 
                    savefigname=None):
    """Show a camera image and the given camera labels.
        Do not draw bboxes that have lower than detection threshold 
    """
        
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    boxnum=len(pred_bbox)
    pred_bbox = [[(i[0], i[1]), (i[2], i[3])] for i in pred_bbox]
    #print(boxnum)
    if len(pred_ids)<1:
        print("No object detected")
        return image
    else:
        
        #print(pred_labels)
        for i in range(boxnum):#patch in pred_bbox:
            if prob[i] > detection_threshold:
                patch=pred_bbox[i]
                #print(patch)
                colorlabel=compute_color_for_labels(pred_ids[i]) #INSTANCE_Color[label]
                #print(colorlabel)#RGB value 0-255
                colorlabelnormalized = [float(i)/255 for i in colorlabel] #0-1
                label=class_dic[pred_ids[i]]
                #print(label)
                ax.add_patch(Rectangle(
                xy=patch[0],#(patch[0], patch[1]), #xmin ymin
                width=patch[1][0]-patch[0][0],#patch[2] - patch[0],
                height=patch[1][1]-patch[0][1],#patch[3] - patch[1],
                linewidth=4,
                edgecolor=colorlabelnormalized,#"red",
                facecolor='none'))
                ax.text(patch[0][0]-5, patch[0][1]-5, label, color=colorlabelnormalized, fontsize=15)
            else:
                continue
        
    ax.imshow(image)
    
    ax.title.set_text(title)
    ax.grid(False)
    ax.axis('off')
    
    if savefigname is not None:
        fig.savefig(savefigname)

def resize(im, img_size=640, square=False):
    """
    https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/inference_video.py
    """
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

def infer_transforms(image):
    """
    https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/inference_video.py
    """
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)

def convert_detections(outputs, detection_threshold, class_dic):
    """
    reference: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/inference_video.py
    Return the bounding boxes, scores, and classes.
    """
    boxes = outputs[0]['boxes'].data.numpy()
    labels = outputs[0]['labels'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()

    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    draw_boxes = boxes.copy()
    # Get all the predicited class names.
    pred_classes = [class_dic[i] for i in outputs[0]['labels'].cpu().numpy()]

    return draw_boxes, pred_classes, scores, labels

def inference_annotations(draw_boxes, pred_classes, scores, labels, classes, colors, orig_image, image):
    """
    https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/inference_video.py
    """
    height, width, _ = orig_image.shape
    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    
    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1 = (int(box[0]/image.shape[1]*width), int(box[1]/image.shape[0]*height))
        p2 = (int(box[2]/image.shape[1]*width), int(box[3]/image.shape[0]*height))
        class_name = pred_classes[j]

        color = colors[labels[j]]
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        
        final_label = class_name + ' ' + str(round(scores[j], 2))
        w, h = cv2.getTextSize(
            final_label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3, 
            thickness=tf
        )[0]  # text width, height
        w = int(w - (0.20 * w))
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(
            orig_image, 
            p1, 
            p2, 
            color=color, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )  
        cv2.putText(
            orig_image, 
            final_label, 
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3.8, 
            color=(255, 255, 255), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return orig_image

#add from https://github.com/lkk688/myyolov7/blob/main/utils/plots.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)