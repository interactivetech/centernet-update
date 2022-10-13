import torch
import torchvision
from torchvision import transforms
import torch.onnx as onnx
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import matplotlib.pyplot as plt
import contextlib
import io
import numpy as np
import cv2
from utils import encode_hm_regr_and_wh_regr
import albumentations as albu
import os
from tqdm import tqdm
from utils import decode_predictions, bbox_xyxy_to_xywh, bbox_xywh_to_xyxy, bbox_iou
def train_transform_norm(annotations,INPUT_SIZE,with_bboxes=True):
    image = annotations['image']
    # print("image.shape: ",image.shape)
    size = (INPUT_SIZE[0], INPUT_SIZE[1]) # height, width
    scale = min(size[0] / image.shape[0], size[1] / image.shape[1])
    intermediate_size = int(image.shape[0] * scale), int(image.shape[1] * scale)
    augmentation = albu.Compose(
        [
            # albu.RandomSizedBBoxSafeCrop(*intermediate_size),
            # albu.Resize(height = intermediate_size[0],width = intermediate_size[1]),
            albu.Resize(*intermediate_size),
            albu.HorizontalFlip(p=0.5),
            albu.HueSaturationValue(p=0.5),
            albu.RGBShift(p=0.5),
            albu.RandomBrightnessContrast(p=0.5),
            albu.MotionBlur(p=0.5),
            albu.PadIfNeeded(*size,border_mode=cv2.BORDER_CONSTANT,mask_value=0.0),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ],
        albu.BboxParams(format='coco', min_area=0.0,
                        min_visibility=0.0, label_fields=['labels'])
    )
    
    augmented = augmentation(**annotations)

    augmented['scale'] = scale    # augmented['image'] = augmented['image'].astype(
    #     np.float32).transpose(2, 0, 1)
    augmented['scale'] = scale

    augmented['in_size'] = image.shape[:2]
    augmented['out_size'] = size
    augmented['intermediate_size'] = intermediate_size
    return augmented

def validation_transform_norm(annotations, INPUT_SIZE,with_bboxes=True):
    bbox_params = None
    if with_bboxes:
        bbox_params = albu.BboxParams(format='coco', min_area=0.0,
                        min_visibility=0.0, label_fields=['labels'])
    
    image = annotations['image']#H,W,C
    # print("image.shape: ",image.shape)

    size = (INPUT_SIZE[0], INPUT_SIZE[1])# h,w
    # print(image.shape)
    scale = min(size[0] / image.shape[0], size[1] / image.shape[1])
    # intermediate_size = [int(dim * scale) for dim in image.shape[:2]]
    intermediate_size = int(image.shape[0] * scale), int(image.shape[1] * scale)

    
    augmentation = albu.Compose(
        [
            albu.Resize(*intermediate_size),
            albu.PadIfNeeded(*size,border_mode=cv2.BORDER_CONSTANT,mask_value=0.0),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ],
        bbox_params
    )
    try:
        augmented = augmentation(**annotations)
    except Exception as e:
        print(annotations)
    augmented['scale'] = scale

    augmented['in_size'] = image.shape[:2]
    augmented['out_size'] = size
    augmented['intermediate_size'] = intermediate_size
    return augmented

class COCODetectionDataset(torch.utils.data.Dataset):

    def __init__(self,
                 img_dir,
                 ann_json,
                 IMG_RESOLUTION=None,
                 MODEL_SCALE=None,
                 transform=None):
        # self.img_id = img_id
        # self.labels = labels
        self.IMG_RESOLUTION = IMG_RESOLUTION
        min_keypoints_per_image = 10
        self.transform = None
        if transform:
            self.transform = transform
        

        with contextlib.redirect_stdout(io.StringIO()):     # redict pycocotools print()
            coco = COCO(ann_json)
        
        # filter no labeled examples
        # ids = []
        # for ds_idx, img_id in enumerate(coco.getImgIds()):
        #     ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #     anno = coco.loadAnns(ann_ids)
        #     if self._has_valid_annotation(anno):
        #         ids.append(ds_idx+1)

        # dataset = torch.utils.data.Subset(dataset, ids)

        cat_ids = sorted(coco.getCatIds())
        label_map = {v: i for i, v in enumerate(cat_ids)}
        print("label_map:", label_map )
        inverse_label_map = {v: k for k, v in label_map.items()}
        img_ids = sorted(coco.getImgIds())
        self.ids = img_ids

        # print(self.ids)
        imgs = coco.loadImgs(self.ids)                       # each img has keys filename, height, width, id
        target = [coco.imgToAnns[idx] for idx in img_ids]   # each ann has keys bbox, category_id, id
        
        img_names = [x["file_name"] for x in imgs]
        targets = []
        # boxes = 
        '''
        (09/03/2022) Andrew: issue with albumentations that cant handle labels with [0,0,0,0]
        if a bbox annotation exists, then do not add the bbox or the label
        '''
        for img_anns, img in zip(target, imgs):
            boxes = []
            labels = []
            for ann in img_anns:
                if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
                    boxes.append(ann['bbox'])
                    labels.append(label_map[ann['category_id']])
            t = {
                'boxes': boxes,
                "labels": labels,
                "image_width":img["width"],
                "image_height":img["height"],
                "image_id":img['id'],
                }
            targets.append(t)
            # {
            # "boxes": [ann["bbox"] for ann in img_anns if ann["bbox"][2] > 0 or ann["bbox"][3] > 0 ],
            # "labels": [label_map[ann["category_id"]] for ann in img_anns],
            # "image_width": img["width"],
            # "image_height": img["height"],
            # "image_id": img["id"]
            # } 
        self.coco = coco
        self.img_dir = img_dir
        self.img_names = img_names
        self.MODEL_SCALE = MODEL_SCALE
        self.targets = targets
        self.transform = transform
        self.num_classes = len(cat_ids)
        self.label_map = label_map
        self.inverse_label_map = inverse_label_map
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # print("idx: ",idx)
        # print(self.img_names[idx])
        # print( self.targets[idx])
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.array(Image.open(img_path).convert("RGB"))

        target = self.targets[idx]
        # print("target['boxes']: ",target['boxes'])
        boxes = np.array(target['boxes'])
        # print("boxes: ", boxes)
        # print("boxes: ",boxes)
        labels = np.array(target['labels'])
        annotations = {'image': img,
                'bboxes':boxes,
                'labels': labels }
        try:
            anns = self.transform(annotations,(self.IMG_RESOLUTION,self.IMG_RESOLUTION))
        except Exception as e:
            print(annotations)

        
        img  = anns['image'].transpose(2,0,1)
        boxes_aug = np.asarray(anns['bboxes'])
        # print("boxes_aug: ",boxes_aug)

        labels = np.asarray(anns['labels'])
        in_size = anns['in_size']
        out_size = anns['out_size']
        intermediate_size = anns['intermediate_size']
        scale =  anns['scale']

        # print("boxes_aug: ",boxes_aug)
        # print("Model Scale: ",self.MODEL_SCALE)
        hm, regr, wh_regr, wh,inds, ind_masks  = encode_hm_regr_and_wh_regr(boxes_aug,
                                                            labels,
                                                            N_CLASSES=self.num_classes,
                                                            input_size=self.IMG_RESOLUTION,
                                                            MODEL_SCALE=self.MODEL_SCALE,
                                                            MAX_N_OBJECTS=128)

        hm = np.ascontiguousarray(hm)
        regr = np.ascontiguousarray(regr)
        wh_regr = np.ascontiguousarray(wh_regr)
        wh = np.ascontiguousarray(wh)
        ind_masks = np.ascontiguousarray(ind_masks)
        inds = np.ascontiguousarray(inds)
            # h_hm_regr_multiclass2(boxesm,reg, cat_wh,cat_reg_mask,inds = make,
            #                                                             labels,
            #                                                             N_CLASSES=2,
            #                                                             input_size=512,
            #                                                             MODEL_SCALE=4,
            #                                                             IN_SCALE=1,
            #                                                             MAX_N_OBJECTS=128)

        return img, hm, regr, wh_regr,wh,ind_masks,inds, in_size, out_size, intermediate_size, scale, boxes_aug, target, idx
        # return img, hm,reg, cat_wh,cat_reg_mask,inds 
def coco_detection_collate_fn(batch):
    img = torch.Tensor(np.stack([x[0] for x in batch], axis=0))
    hm = torch.Tensor(np.stack([x[1] for x in batch], axis=0))
    regr = torch.Tensor(np.stack([x[2] for x in batch], axis=0))
    wh = torch.Tensor(np.stack([x[3] for x in batch], axis=0))
    ind_masks = torch.Tensor(np.stack([x[4] for x in batch], axis=0))
    inds = torch.Tensor(np.stack([x[5] for x in batch], axis=0)).type(torch.int64)
    in_size = torch.Tensor(np.stack([x[6] for x in batch], axis=0))
    out_size = torch.Tensor(np.stack([x[7] for x in batch], axis=0))
    intermediate_size = torch.Tensor(np.stack([x[8] for x in batch], axis=0))
    scale = torch.Tensor(np.stack([x[9] for x in batch], axis=0))
    # print(np.vstack([x[10] for x in batch]).shape)
    boxes_aug = torch.Tensor(np.vstack([x[10] for x in batch]))
    targets = [x[11] for x in batch]
    idxs = [x[12] for x in batch]

    return img, hm,regr,wh_regr, wh,ind_masks,inds, in_size, out_size, intermediate_size, scale, boxes_aug, targets, idxs

if __name__ == '__main__':
    # TODO, test and make sure COCO dataset is loading correctly
    # /run/determined/workdir/coco_dataset/annotations/instances_minitrain2017.json
    # /run/determined/workdir/coco_dataset/train2017
    # /run/determined/workdir/coco_dataset/train2017
    DATASET_DIR = '/run/determined/workdir/coco_dataset'
    IMG_RESOLUTION=256 
    MODEL_SCALE=1
    # Mini COCO Dataset
    ds = COCODetectionDataset(os.path.join(DATASET_DIR,'train2017'),
    os.path.join(DATASET_DIR,'annotations/instances_minitrain2017.json'),
    transform=train_transform_norm,
    MODEL_SCALE=MODEL_SCALE,
    IMG_RESOLUTION=IMG_RESOLUTION)
    val_ds = COCODetectionDataset(os.path.join(DATASET_DIR,'val2017'),
    os.path.join(DATASET_DIR,'annotations/instances_val2017.json'),
    transform=train_transform_norm,
    MODEL_SCALE=MODEL_SCALE,
    IMG_RESOLUTION=IMG_RESOLUTION)
                                  
    for ind, (img,
                  hm, 
                  regr, 
                  wh_regr,
                  wh,
                  inds_mask,
                  inds,
                  in_size,
                  out_size,
                  intermediate_size,
                  scale,
                  boxes_aug,
                  target,
                  idxs) in enumerate(tqdm(ds)):
        # print(hm.shape)
        # print(regr.shape)
        # print(wh_regr.shape)
        # hm, regr, wh_regr, w_h_,inds, ind_masks= encode_hm_regr_and_wh_regr(boxes_o, classes,N_CLASSES=2, input_size=128,MODEL_SCALE=2)
        # print(hm.shape)
        # print("hm.shape: {}, regr.shape: {}, wh_regr.shape: {}, w_h_.shape: {},inds.shape: {}, ind_masks.shape: {}".format(hm.shape, 
        #                                                                                             regr.shape, 
        #                                                                                             wh_regr.shape, 
        #                                                                                             w_h_.shape,
        #                                                                                             inds.shape, 
        #                                                                                             ind_masks.shape))
        hm_b = torch.from_numpy(hm).unsqueeze(0)
        regr_b = torch.from_numpy(regr).unsqueeze(0)
        # print(hm_b.shape)
        # print(regr_b.shape)
        
        wh_regr_b = torch.from_numpy(wh_regr).unsqueeze(0)
        # print(wh_regr_b.shape)
        boxes,scores,clses = decode_predictions(hm_b,
                                                regr_b,
                                                wh_regr_b,
                                                MODEL_SCALE=MODEL_SCALE,
                                                K=100,
                                                nms_thresh=0.5)
        '''
        boxes = np.array([[44,44,76,76],
                        [15,17,113,113],
                        [38,38,70,70]])
        '''
        # print(boxes_o)
        print(boxes.shape, boxes_aug.shape)
        print(bbox_xyxy_to_xywh(boxes.numpy()),scores,clses)
        gt_bbox = bbox_xywh_to_xyxy(boxes_aug)
        print(bbox_iou(gt_bbox, boxes.numpy()))
        break