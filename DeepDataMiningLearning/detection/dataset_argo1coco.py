#ref: https://github.com/lkk688/WaymoObjectDetection/blob/master/MyDetector/torchvision_waymococo_train.py
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from glob import glob
import os
import math
import itertools
import torch.utils.data as data
from pycocotools.coco import COCO

class Argo1COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.coco = COCO(annotation)
        self.is_train = train
        self.ids = list(sorted(self.coco.imgs.keys())) #id string list

        #
        dataset=self.coco.dataset #'images': image filename (images/xxx.jpg) with image_id (0000001)
        imgToAnns=self.coco.imgToAnns #image_id to list of annotations
        catToImgs =self.coco.catToImgs #three classes, 1,2,4
        cats=self.coco.cats

        #num_classes=5 # ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
        #previous_num_classes = 4 #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
        #Real data only has 
        self.INSTANCE_CATEGORY_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign']
        self.numclass = len(self.INSTANCE_CATEGORY_NAMES)
        #self.INSTANCE2id = {'__background__':0, 'Vehicles': 1, 'Pedestrians': 2, 'Cyclists': 4} #background is 0
        #self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        #In annotation, class is 1,2,4

    
    def _get_target(self, id):
        'Get annotations for sample'

        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=id) # 1 or 2 ...
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py    
        annotations = self.coco.loadAnns(ann_ids) 

        # if id == 1:
        #     print(f"annotations: {annotations}")
        #     print(f"len: {len(annotations)}")

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            categories.append(cat)

        target = (torch.FloatTensor(boxes),
                  torch.FloatTensor(categories).unsqueeze(1))

        return target


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        imginfo=self.coco.imgs[img_id]
        path_train_or_val = 'train/' if self.is_train else 'val/'
        path = path_train_or_val + imginfo['file_name'] 
        #print(f'index: {index}, img_id:{img_id}, info: {imginfo}')

        # path for input image
        #loadedimglist=coco.loadImgs(img_id)
        # print(loadedimglist)
        #path = coco.loadImgs(img_id)[0]['file_name']
        #print("image path:", path)
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        #img = Image.open(os.path.join(self.root, path)).convert('RGB')

        
        if index == 1:
            display(img)
            print(f'image info: {path}')

        # List: get annotation id from coco
        #ann_ids = coco.getAnnIds(imgIds=img_id)
        annolist=[self.coco.imgToAnns[img_id]]
        anns = list(itertools.chain.from_iterable(annolist))
        ann_ids = [ann['id'] for ann in anns]
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        targets  = coco.loadAnns(ann_ids)
        #targets=self.anns[ann_ids]
        #print("targets:", targets)
        
        #image_id = targets["image_id"].item()

        # number of objects in the image
        num_objs = len(targets)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        target = {}
        target_bbox = []
        target_labels = []
        target_areas = []
        target_crowds = []
        for i in range(num_objs):
            xmin = targets[i]['bbox'][0]
            ymin = targets[i]['bbox'][1]
            width = targets[i]['bbox'][2]
            xmax = xmin + width
            height = targets[i]['bbox'][3]
            ymax = ymin + height
            if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0 and width>1 and height>1:
                target_bbox.append([xmin, ymin, xmax, ymax])
                target_labels.append(targets[i]['category_id'])
                target_crowds.append(targets[i]['iscrowd'])
                target_areas.append(targets[i]['area'])
        num_bbox=len(target_bbox)
        assert num_objs == num_bbox
        #print("target_bbox len:", num_objs)
        if num_objs>0:
            #print("target_labels:", target_labels)
            target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
            # Labels int value for class
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
            #target['image_id'] = torch.tensor([int(img_id)])
            #target['image_id'] = torch.tensor(int(img_id))
            target['image_id'] = int(img_id)
            #torch.tensor([int(frameitem.context.name.split("_")[-2] + str(index))])
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#torch.zeros((len(target['boxes'])), dtype=torch.int64)
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)#empty
            #target['image_id'] = torch.tensor([int(img_id)])
            #target['image_id'] = torch.tensor(int(img_id))
            target['image_id'] = int(img_id)
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty

        if self.transform:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

import DeepDataMiningLearning.detection.transforms as T
def get_transformsimple():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    data_root = '/data/cmpe249-fa23/Argoverse/Argoverse-1.1/images/'
    ann_file = '/data/cmpe249-fa23/argo1COCO/argo1_10K.json'
    myargo1coco = Argo1COCODataset(root=data_root,  
                            annotation=ann_file,
                            transform=get_transformsimple())
    length = len(myargo1coco)
    print("Dataset",length)#85008
    img, target = myargo1coco[0]
    print(target.keys())