import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from DeepDataMiningLearning.detection.coco_utils import get_coco
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import csv
from DeepDataMiningLearning.detection.dataset_kitti import KittiDataset
from DeepDataMiningLearning.detection.dataset_waymococo import WaymoCOCODataset
from DeepDataMiningLearning.detection.dataset_argo1coco import Argo1COCODataset
from collections import defaultdict

import torch
import DeepDataMiningLearning.detection.transforms as reference_transforms


WrapNewDict = False

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors

        return torchvision.transforms.v2, torchvision.tv_tensors
    else:
        return reference_transforms, None


class DetectionPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter.
    def __init__(
        self,
        *,
        data_augmentation,
        hflip_prob=0.5,
        mean=(123.0, 117.0, 104.0),
        backend="pil",
        use_v2=False,
    ):

        T, tv_tensors = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if data_augmentation == "hflip":
            transforms += [T.RandomHorizontalFlip(p=hflip_prob)]
        elif data_augmentation == "lsj":
            transforms += [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                # TODO: FixedSizeCrop below doesn't work on tensors!
                reference_transforms.FixedSizeCrop(size=(1024, 1024), fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            fill = defaultdict(lambda: mean, {tv_tensors.Mask: 0}) if use_v2 else list(mean)
            transforms += [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=fill),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2.
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [
                T.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                T.SanitizeBoundingBoxes(),
                T.ToPureTensor(),
            ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self, backend="pil", use_v2=False):
        T, _ = get_modules(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]
        elif backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        else:
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_dataset(datasetname, is_train, is_val, args):
    if datasetname.lower() == 'coco':
        ds, num_classes = get_cocodataset(is_train, is_val, args)
    elif datasetname.lower() == 'kitti':
        ds, num_classes = get_kittidataset(is_train, is_val, args)
    elif datasetname.lower() == 'waymococo':
        ds, num_classes = get_waymococodataset(is_train, is_val, args)
    elif datasetname.lower() == 'argo1coco':
        ds, num_classes = get_argo1cocodataset(is_train, is_val, args)

    return ds, num_classes

def get_transform(is_train, args):
    if is_train:
        return DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)
    
def get_cocodataset(is_train, is_val, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes

   
def get_kittidataset(is_train, is_val, args):
    rootPath=args.data_path
    #dataset = MyKittiDetection(rootPath, train=True, transform=get_transform(is_train, args))
    if is_val == True:
        transformfunc=get_transform(False, args)
        dataset = KittiDataset(rootPath, train=True, split='val', transform=transformfunc)
    else:
        transformfunc=get_transform(True, args) #add augumentation
        dataset = KittiDataset(rootPath, train=is_train, split='train', transform=transformfunc)
    
    num_classes = dataset.numclass
    return dataset, num_classes
    #mykitti = datasets.Kitti(root=rootPath, train= True, transform = get_transform(is_train, args), target_transform = None, download = False)

def get_waymococodataset(is_train, is_val, args):
    rootPath=args.data_path
    annotation=args.annotationfile
    if is_val == True:
        annotation = os.path.join(rootPath, 'annotations_val20new.json') #'annotations_val50new.json'
        transformfunc=get_transform(False, args)
        dataset = WaymoCOCODataset(rootPath, annotation, train=True, transform=transformfunc)
    else: #Training
        annotation = os.path.join(rootPath, 'annotations_train200new.json') 
        transformfunc=get_transform(True, args) #add augumentation
        dataset = WaymoCOCODataset(rootPath, annotation, train=is_train, transform=transformfunc)
    
    num_classes = dataset.numclass
    return dataset, num_classes
    #mykitti = datasets.Kitti(root=rootPath, train= True, transform = get_transform(is_train, args), target_transform = None, download = False)

def get_argo1cocodataset(is_train, is_val, args):
    rootPath=args.data_path
    annotation_path=args.annotationfile
    if is_val == True:
        annotation = os.path.join(annotation_path, 'argo1_val_all.json') #'annotations_val50new.json'
        transformfunc=get_transform(False, args)
        dataset = Argo1COCODataset(rootPath, annotation, train=is_train, transform=transformfunc)
    else: #Training
        annotation = os.path.join(annotation_path, 'argo1_all.json') 
        transformfunc=get_transform(True, args) #add augumentation
        dataset = Argo1COCODataset(rootPath, annotation, train=is_train, transform=transformfunc)
    
    num_classes = dataset.numclass
    return dataset, num_classes

