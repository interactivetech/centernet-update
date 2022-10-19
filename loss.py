
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _tranpose_and_gather_feature, _gather_feature
from model import EfficientCenterDet
from tqdm import tqdm
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
import os
def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  # print("regr: ",regr)
  # print("gt_regr: ",gt_regr)
  # print("mask: ",mask.shape)
  num = mask.float().sum()
#   print("")
  # print("mask: ", mask.shape)
  # print("mask.unsqueeze(2): ", mask.unsqueeze(2).shape)
  # print("mask.unsqueeze(2).expand_as(gt_regr): ", mask.unsqueeze(2).expand_as(gt_regr).shape)
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()
#   mask = mask.float()
#   print("regr: ",regr.shape)
#   print("mask: ",mask.shape)
  regr = regr * mask
  gt_regr = gt_regr * mask
#   print("regr: ",regr.shape)
#   print("gt_regr: ",gt_regr)
  regr_loss = nn.functional.l1_loss(regr, gt_regr, reduction='sum')
#   print("regr_loss: ",regr_loss)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss


def neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x h x w)
      gt (batch  x h x w)
  '''
#   pred = pred.unsqueeze(1)float()
#   gt = gt.unsqueeze(1).float()

  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)
  neg_weights = torch.pow(1 - gt[neg_inds], 4)
  
  loss = 0
  # prob_pred = F.logsigmoid(pred)
  # pos_loss = F.logsigmoid(pred) * torch.pow(1 - prob_pred, 3) * pos_inds
  # neg_loss = F.logsigmoid(1 - pred) * torch.pow(prob_pred, 3) * neg_inds *  neg_weights 
  ppred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]
  # pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) *neg_weights
  # print("neg_inds: ",neg_inds.sum())
  # print("neg_weights: ",neg_weights.sum())
  # print("torch.pow(pred, 2)*torch.log(1 - pos_pred): ",(torch.pow(pos_pred, 2)*torch.log(1 - pos_pred)).sum())
  # print("torch.log(1 - neg_pred) * torch.pow(neg_pred, 2)*neg_weights: ", (torch.log(1 - neg_pred) * torch.pow(neg_pred, 2)*neg_weights).sum())

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  print("num_pos: ",num_pos)
  print("pos_loss: ",pos_loss)
  print("neg_loss: ",neg_loss)
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss =  loss - (pos_loss + neg_loss) / num_pos
  return loss

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pred = torch.sigmoid(pred)
  # print("min: ",gt.min(),"max: ",gt.max())
  # gt = torch.sigmoid(gt)
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
  # print("torch.log(1 - pred) * torch.pow(pred, 2): ",(torch.log(1 - pred) * torch.pow(pred, 2)).sum() )
  # print("torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights: ",(torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights).sum())

  num_pos  = pos_inds.float().sum()
  # print("num_pos: ", num_pos)
  pos_loss = pos_loss.sum()
  # print("pos_loss: ", pos_loss)
  neg_loss = neg_loss.sum()
  # print("neg_loss: ", neg_loss)
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss
def centerloss(pred_hm, 
               hm,
               pred_regr, 
               regr,
               pred_wh_regr, 
               wh_regr,
               mask,
               inds,
               wh,
               regr_,
               weight=0.4, size_average=True):
    # Binary mask loss
    '''
    pred_hm: [4,1,128, 128]
    hm: [4,1,128, 128]
    pred_regr: (4,2,128,128)
    reg: (4,2,128,128)
    '''
    # print("pred_hm:" ,pred_hm.shape)
    # print("hm: ",hm.shape)
    # print("pred_regr: ",pred_regr.shape)
    # print("regr: ",regr.shape)
    # print("pred_wh_regr: ",pred_wh_regr.shape)
    # print("wh_regr: ",wh_regr.shape)
    # print("mask: ",mask.shape)
    # print("inds: ",inds.shape)
    # print("wh: ",wh.shape)
    # print("regr_: ",regr_.shape)
    # pred_mask = prediction[:, 0]# torch.Size([4, 128, 128])
    # print("pred_mask: ",pred_mask.shape)

    # mask_loss = neg_loss2(pred_hm, hm)
    mask_loss = _neg_loss(pred_hm,hm)
    # mask = hm>0.99  
    # Regression L1 loss
    pred_regr_m = _tranpose_and_gather_feature(pred_regr, inds)
    regr_loss = _reg_loss(pred_regr_m,regr_,mask)

    pred_wh_regr_m = _tranpose_and_gather_feature(pred_wh_regr, inds)
    regr_wh_loss = 0.1*_reg_loss(pred_wh_regr_m,wh,mask)

    loss =mask_loss + regr_loss + regr_wh_loss
    return loss ,mask_loss , regr_loss, regr_wh_loss

if __name__ == '__main__':
    
    DATASET_DIR = '/run/determined/workdir/coco_dataset'
    IMG_RESOLUTION=256 
    MODEL_SCALE=1
    # Mini COCO Dataset
    ds = COCODetectionDataset(os.path.join(DATASET_DIR,'train2017'),
    os.path.join(DATASET_DIR,'annotations/instances_minitrain2017.json'),
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
                  regr_,
                  in_size,
                  out_size,
                  intermediate_size,
                  scale,
                  boxes_aug,
                  target,
                  idxs) in enumerate(tqdm(ds)):

        hm_b = torch.from_numpy(hm).unsqueeze(0)
        regr_b = torch.from_numpy(regr).unsqueeze(0)
        wh_regr_b = torch.from_numpy(wh_regr).unsqueeze(0)
        inds_mask = torch.from_numpy(inds_mask).unsqueeze(0)
        inds = torch.from_numpy(inds).unsqueeze(0)
        wh = torch.from_numpy(wh).unsqueeze(0)
        regr_ = torch.from_numpy(regr_).unsqueeze(0)
        # print("wh: ",wh)
        # pred_wh_regr_m = _tranpose_and_gather_feature(wh_regr_b, inds)

        # print("pred_wh_regr_m :", pred_wh_regr_m)
        # r_loss = _reg_loss(wh,pred_wh_regr_m,inds_mask)
        # print("r_loss :", r_loss)
        loss ,mask_loss , regr_loss, regr_wh_loss = centerloss(hm_b,
                                                                   hm_b,
                                                                   regr_b,
                                                                   regr_b,
                                                                   wh_regr_b,
                                                                   wh_regr_b,
                                                                   inds_mask,
                                                                   inds,
                                                                   wh,
                                                                   regr_,
                                                                   weight=0.4, 
                                                                   size_average=True)
        print("Loss: ",loss)
        print("mask_loss: ",mask_loss)
        print("regr_loss: ",regr_loss)
        print("regr_wh_loss: ",regr_wh_loss)
        break
    

