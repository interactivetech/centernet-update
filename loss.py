
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _transpose_and_gather_feat, _gather_feat

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
#   print("")
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

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()
  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  # prob_pred = F.logsigmoid(pred)
  # pos_loss = F.logsigmoid(pred) * torch.pow(1 - prob_pred, 3) * pos_inds
  # neg_loss = F.logsigmoid(1 - pred) * torch.pow(prob_pred, 3) * neg_inds *  neg_weights 
  pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 3) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 3) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss =  loss - (pos_loss + neg_loss) / (num_pos+1e-4)
  return loss
def centerloss(pred_hm, hm,pred_regr, regr,mask,inds,wh,weight=0.4, size_average=True):
    # Binary mask loss
    '''
    pred_hm: [4,1,128, 128]
    hm: [4,1,128, 128]
    pred_regr: (4,2,128,128)
    reg: (4,2,128,128)
    '''
    # pred_mask = prediction[:, 0]# torch.Size([4, 128, 128])
    # print("pred_mask: ",pred_mask.shape)

    # mask_loss = neg_loss2(pred_hm, hm)
    mask_loss = neg_loss(pred_hm,hm)
    # mask = hm>0.99  
    # Regression L1 loss
    pred = _transpose_and_gather_feat(pred_regr, inds)
    # print("pred: ",pred)
    # print("wh: ",wh)
    regr_loss = _reg_loss(pred,wh,mask)

    loss =mask_loss + regr_loss   
    return loss ,mask_loss , regr_loss
