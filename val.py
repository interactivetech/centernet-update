# ToDo: Do batched validation, make it fast!
from unicodedata import decimal
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from utils import encode_hm_regr_and_wh_regr, decode_predictions, rescale_boxes, per_class_coco_ap, per_class_coco_ap50
from loss import centerloss
import torch
import cv2
import torch
from PIL import Image
# from val import val
from model import EfficientCenterDet
import os
import json
from pycocotools.cocoeval import COCOeval
from compute_map import compute_map, index_pred_and_gt_by_class
from time import time
def compute_non_coco(cls_names,pred, gt,epoch,writer):
    '''
    '''
    print("Computing mAP (Non COCO)...")
    t0 = time()
    maps = []
    maps_50 = []
    for c_ind,c in enumerate(tqdm(cls_names)):
            p,g = index_pred_and_gt_by_class(pred,gt,c_ind)
            mAP = compute_map(g,p,c_ind)
            maps.append(mAP[c_ind])
            maps_50.append(mAP[str(c_ind)+'_50'])
            print("{}-mAP: {}".format(c,mAP[c_ind]))
            print("{}-mAP_50: {}".format(c,mAP[str(c_ind)+'_50']))
            if writer is not None:
                    writer.add_scalar("{} mAP/val".format(c), mAP[c_ind], epoch)
                    writer.add_scalar("{} mAP/val".format(c+'_50'), mAP[str(c_ind)+'_50'], epoch)
    print("mAP calculation completed! Time: {}".format(time() - t0))
    mean_mAP = np.array(maps).mean()
    mean_mAP_50 = np.array(maps_50).mean()
    print("mean mAP: {}".format(mean_mAP))
    print("mean mAP 50: {}".format(mean_mAP_50))
    if writer is not None:
            writer.add_scalar("mAP/val".format(c), mean_mAP, epoch)
            writer.add_scalar("mAP_50/val".format(c), mean_mAP_50, epoch)
def torch_to_numpy(bboxes,scores,classes):
    if torch.cuda.is_available():
        bboxes = bboxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        classes=classes.cpu().detach().numpy()
    else:
        b = b.detach().numpy()
        scores = scores.numpy()
        classes=classes.numpy()
    return bboxes,scores,classes
def compute_coco_metrics(detections,val_ds):
    '''
    '''
    results_per_category = None
    results_per_category_50 = None
    cls_names = []
    if len(detections) > 0:
        json.dump(detections, open('res.json', 'w'))
        coco_dets = val_ds.coco.loadRes('res.json')
        coco_eval = COCOeval(val_ds.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # model.eval()
        results_per_category = per_class_coco_ap(val_ds.coco,coco_eval)
        results_per_category_50 = per_class_coco_ap50(val_ds.coco,coco_eval)
        # print(results_per_category)
        # [('circle', '0.006'), ('rectangle', '0.001')]
        
        for i in results_per_category:
            cls_name = i[0]
            cls_names.append(i[0])
            mAP_val = float(i[1])
            # print(cls_name,mAP_val)
    return results_per_category, results_per_category_50
def convert_predictions_to_coco(image_id,bboxes,scores,classes):
    '''
    '''
    pred = {}
    pred[image_id] = {
        'boxes': [],
        'scores': [],
        'classes': []
    }
    detections = []

    for ind,(b,s,c) in enumerate(zip(bboxes,scores,classes)):
        '''
        det[1:5].astype(np.float64), decimals=2).tolist(),
            'score': float(np.around(det[5], decimals=3)),
        '''
        # x,y,x2,y2 = np.around(b,decimals=2).tolist()
        x,y,x2,y2 = b.tolist()

        # print([int(x),int(y),int(x2),int(y2)])
        # print(int(c))
        pred[image_id]['boxes'].append([int(x),int(y),int(x2),int(y2)])
        pred[image_id]['scores'].append(float(s))
        pred[image_id]['classes'].append(int(c))
        # print("classes: ",c)
        # print(" x,y,x2,y2: ", x,y,x2,y2)
        w = x2-x
        h = y2-y
        # print(" x,y,w,h: ", x,y,w,h)

        # area = abs(w*h)
        # print("pred: ", [int(x),int(y),int(w),int(h)])
        # print(image_id, b, s, c,area)
        # print(float(s))
        detection = {
                'image_id':image_id,
                'bbox': [round(x, 2),round(y, 2),round(w, 2),round(h, 2)],
                'category_id': int(c),
                'score': float(s),
                'area': w*h
            }
        # print(detection)
        detections.append(detection)
    return detections, pred

def val(model,
        val_ds,
        val_loader,
        writer,
        epoch,
        visualize_res=None,
        IMG_RESOLUTION=None,
        device=None,
        MODEL_SCALE=None):
    '''
    '''
    model.eval()
    detections_total = []
    gt = {}
    preds = {}
    for img,hm, regr, wh_regr,wh,inds_mask,inds,regr_,in_size, out_size, intermediate_size, scale, boxes_aug, target, idxs in tqdm(val_loader):
        img = img.to(device)
        hm = hm.to(device)
        regr = regr.to(device)
        wh_regr = wh_regr.to(device)
        wh = wh.to(device)
        regr_ = regr_.to(device)
        inds_mask = inds_mask.to(device)
        inds = inds.to(device)
        for t in target:
                gt[t['image_id']]= {
                        'boxes':[[int(i[0]),int(i[1]),int(i[0])+int(i[2]),int(i[1])+int(i[3])] for i in  t['boxes']],
                        'classes':t['labels']
                        }
        # print("inds_mask: ", inds_mask.shape)
        # print("inds: ", inds.shape)
        # optimizer.zero_grad()
        # print("begin")
        image_ids = [i['image_id'] for i in target]
        bboxes_gt = np.vstack([np.array(i['boxes']).reshape(-1,4) for i in target])
        with torch.no_grad():
            pred_hm, pred_regs, pred_wh_regs = model(img)
            pred_hm = torch.sigmoid(pred_hm)
            # print("pred_hm.shape[0]: ",pred_hm.shape[0])
            for ind in range(pred_hm.shape[0]):# batch size for val is always 1,C,H,W
                                image_id = image_ids[ind]
                                boxes,scores,clses = decode_predictions(pred_hm[ind].unsqueeze(0),
                                                pred_regs[ind].unsqueeze(0),
                                                pred_wh_regs[ind].unsqueeze(0),
                                                MODEL_SCALE=MODEL_SCALE,
                                                K=100,
                                                nms_thresh=0.5)
                                boxes,scores,clses = torch_to_numpy(boxes,scores,clses)
                                # print("boxes: ",boxes[0])
                                boxes = rescale_boxes( boxes, out_size[ind].numpy(), intermediate_size[ind].numpy(), scale[ind].numpy())

                                # print("boxes: ",boxes[0])
                                # print("image_id: ",image_id)
                                detections, pred = convert_predictions_to_coco(image_id,boxes, scores, clses)
                                detections_total+=detections
                                preds.update(pred)
                                # print("pred[{}]: {}".format(image_id,pred[image_id]))
                                # print("preds: ",preds.keys())
        print("Len of Detections: ",len(detections_total))
    results_per_category, results_per_category_50s = compute_coco_metrics(detections_total,val_ds)
    if results_per_category is not None and results_per_category_50s is not None:
        # pred_hm
        hm_pred2 = pred_hm[0].cpu().numpy().sum(0)
        h,w = hm_pred2.shape
        # print("hm_pred2: ", hm_pred2.shape)
        hm_pred = np.dstack([hm_pred2*255]*3).astype(np.uint8)
        print("hm_pred: ",hm_pred.shape)
        if writer is not None:
            writer.add_image('hm_pred_{}'.format(0), hm_pred, epoch, dataformats='HWC')
        print("results_per_category: ",results_per_category)
        # visualize pred_hm
        cls_names = []
        average = 0
        for i in results_per_category:
            cls_name = i[0]
            cls_names.append(i[0])
            mAP_val = float(i[1])
            print("map: ",cls_name,mAP_val)
            average+=mAP_val
            if writer is not None:
                writer.add_scalar("{} coco mAP/val".format(cls_name), mAP_val, epoch)
        average /=len(cls_names)
        print("average map: ",average)
        if writer is not None:
            writer.add_scalar("coco mAP/val", average, epoch)
        average = 0
        for i in results_per_category_50s:
            cls_name = i[0]
            mAP_50_val = float(i[1])
            print("map50: ",cls_name,mAP_50_val)
            average+=mAP_50_val
            if writer is not None:
                writer.add_scalar("{} coco mAP50/val".format(cls_name), mAP_50_val, epoch)
        average /=len(cls_names)
        print("average map50: ",average)
        if writer is not None:
            writer.add_scalar("coco mAP50/val", average, epoch)
        compute_non_coco(cls_names,preds,gt,epoch,writer)
    else:
        print("No COCO MAP, len detections: {}".format(detections_total))
        #             # print(len(target)):




if __name__ == '__main__':
    DATASET_DIR = '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/'
    IMG_RESOLUTION=256 
    MODEL_SCALE=2
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
    

    BATCH_SIZE = 48

    train_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=8,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn)
    
    # LR = 1e-2
    LR = 1e-3
    # LR = 2.5e-4*BATCH_SIZE
    from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(comment='mv2')
    # writer = SummaryWriter(comment='_mini_coco_emv2')
    writer = None


    multi_gpu=False
    # visualize_res=IMG_RESOLUTION//4
    visualize_res=IMG_RESOLUTION//MODEL_SCALE
    model = EfficientCenterDet(val_ds.num_classes)
    model.load_state_dict(torch.load('/home/projects/centernet-update/ddp_efficient_centernet_150.pth'))
    model.to('cuda:0')
    val(model,val_ds,val_loader,writer,0,visualize_res=None,IMG_RESOLUTION=None,device='cuda:0')