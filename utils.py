import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.ops import nms, box_convert

def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (x, y, w, h).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return x1, y1, w, h
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def bbox_clip_xyxy(xyxy, width, height):
    """Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.

    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    width : int or float
        Boundary width.
    height : int or float
        Boundary height.

    Returns
    -------
    type
        Description of returned object.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return x1, y1, x2, y2
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))
def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  # print(gaussian.shape)
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap
def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)

  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value

  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]

  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap

def encode_hm_regr_and_wh_regr(boxes, classes,N_CLASSES=None, input_size=None,MODEL_SCALE=None,MAX_N_OBJECTS=None):
    '''
    '''
    boxes = bbox_xywh_to_xyxy(boxes)
    max_objs=MAX_N_OBJECTS
    fmap_dim = input_size//MODEL_SCALE
    hm = np.zeros((N_CLASSES,fmap_dim,fmap_dim),dtype=np.float32)
    regr = np.zeros((2, fmap_dim, fmap_dim), dtype=np.float32)
    wh_regr = np.zeros((2, fmap_dim, fmap_dim), dtype=np.float32)
    inds = np.zeros((max_objs,), dtype=np.int64)
    ind_masks = np.zeros((max_objs,), dtype=np.uint8)
    w_h_ = np.zeros((max_objs, 2), dtype=np.float32)  # width and height
    regr_ = np.zeros((max_objs, 2), dtype=np.float32)  # centernet regression
    for ind,(c,cl) in enumerate(zip(boxes,classes)):
        x,y,x2,y2 = c
        w = int(x2-x)
        h = int(y2-y)
        # print(w,h)
        centers = [(x+x2)/2.,(y+y2)/2.]
        # print("centers: ",centers)
        c_s = np.array([i/MODEL_SCALE for i in centers])
        # print("c_s: ",c_s)
        centers_int = c_s.copy().astype(np.int32)
        # print("centers_int: ",centers_int)
        # centers_sc = np.array([int(i)//MODEL_SCALE for i in c_s],dtype=np.float32)
        # print("centers-centers_int: ",centers-centers_int)
        radius = gaussian_radius((math.ceil(h//MODEL_SCALE), math.ceil(w//MODEL_SCALE)),min_overlap=1.0)
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[cl], centers_int, 
                                radius)
        draw_dense_reg(wh_regr,hm[cl],c_s,np.array([w,h],dtype=np.float32),radius)
        draw_dense_reg(regr,hm[cl],c_s,centers-centers_int,radius)
        w_h_[ind] = np.array([1.*w,1.*h],dtype=np.float32)
        regr_[ind]=centers-centers_int
        inds[ind] = centers_int[1] * fmap_dim + centers_int[0]
        ind_masks[ind] = 1
    
    return hm, regr, wh_regr, w_h_,inds, ind_masks, regr_

def visualize_gt_on_output(boxes, classes,hm,MODEL_SCALE=None):
    '''
    '''
    hm_vis = hm.sum(0)
    for b,cl in zip(boxes, classes):
        x,y,x2,y2 = b
        w = int(x2-x)
        h = int(y2-y)
        centers = [(x+x2)/2,(y+y2)/2]
        c_s = np.array([int(i//MODEL_SCALE) for i in centers]).astype(np.int32)
        print(c_s[0]-w//MODEL_SCALE//2)
        cv2.rectangle(hm_vis,(c_s[0]-w//MODEL_SCALE//2,c_s[1]-h//MODEL_SCALE//2),(c_s[0]+w//MODEL_SCALE//2,c_s[1]+h//MODEL_SCALE//2),1)
    return hm_vis

def nonempty(boxes,threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.
        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

def _gather_feature(feat, ind, mask=None):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  if mask is not None:
    mask = mask.unsqueeze(2).expand_as(feat)
    feat = feat[mask]
    feat = feat.view(-1, dim)
  return feat


def _tranpose_and_gather_feature(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feature(feat, ind)
  return feat

def _topk(scores, K=40):
  batch, cat, height, width = scores.size()

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def decode_predictions(hm,regr,wh_regr,MODEL_SCALE=None,K=100,nms_thresh=None):
    '''
    hm: BxCxHxW
    regr: Bx2xHxW
    wh_regr: Bx2xHxW
    '''
    
    batch,cat, height, width = hm.shape

    scores, inds, clses, ys, xs = _topk(hm, K=K)
    # print("inds: ",inds.shape)
    # print("regr: ",regr.shape)
    regs = _tranpose_and_gather_feature(regr, inds)
    regs = regs.view(batch, K, 2)
    # print(regs.shape)
    # print(xs.view(batch, K, 1))
    # print(xs.view(batch, K, 1)*MODEL_SCALE)
    # print(xs.view(batch, K, 1)*MODEL_SCALE+regs[:, :, 0:1])
    xs = xs.view(batch, K, 1)+regs[:, :, 0:1]
    ys = ys.view(batch, K, 1)+regs[:, :, 1:2]
    # print(xs)


    w_h_ = _tranpose_and_gather_feature(wh_regr, inds)
    w_h_ = w_h_.view(batch, K, 2)
    # print(ys)
    # print(w_h_ /2)
    clses = clses.view(batch, K, 1).float().squeeze(0).squeeze(-1)
    scores = scores.view(batch, K, 1).squeeze(0).squeeze(-1).type(torch.float32)
    boxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                      ys - w_h_[..., 1:2] / 2,
                      xs + w_h_[..., 0:1] / 2,
                      ys + w_h_[..., 1:2] / 2], dim=2).squeeze(0)

    keep = nonempty(boxes)
    boxes=boxes[keep]
    scores=scores[keep]
    clses = clses[keep]
    idx = nms(boxes, scores, nms_thresh)
    boxes = boxes[idx]
    scores = scores[idx]
    clses = clses[idx]
    
    return boxes,scores,clses

if __name__ == '__main__':
    input_size = 128
    MODEL_SCALE=2
    # cs = np.array([[60,60,32,32],[64,64,96,96],[54,54,32,32]])
    boxes_o = np.array([[44,44,76,76],
                    [16,18,113,113],
                    [38,38,70,70]])

    classes = np.array([0,1,0])
    hm, regr, wh_regr, w_h_,inds, ind_masks, regr_= encode_hm_regr_and_wh_regr(boxes_o, 
                                                                               classes,
                                                                               N_CLASSES=2, 
                                                                               input_size=128,
                                                                               MODEL_SCALE=2,
                                                                               MAX_N_OBJECTS=128)
    print(hm.shape)
    print("hm.shape: {}, regr.shape: {}, wh_regr.shape: {}, w_h_.shape: {},inds.shape: {}, ind_masks.shape: {}, regr_: {}".format(hm.shape, 
                                                                                                regr.shape, 
                                                                                                wh_regr.shape, 
                                                                                                w_h_.shape,
                                                                                                inds.shape, 
                                                                                                ind_masks.shape,
                                                                                                regr_.shape))
    hm_b = torch.from_numpy(hm).unsqueeze(0)
    regr_b = torch.from_numpy(regr).unsqueeze(0)
    wh_regr_b = torch.from_numpy(wh_regr).unsqueeze(0)
    boxes,scores,clses = decode_predictions(hm_b,regr_b,
                                            wh_regr_b,
                                            MODEL_SCALE=MODEL_SCALE,
                                            K=100,
                                            nms_thresh=0.5)
    '''
    boxes = np.array([[44,44,76,76],
                    [15,17,113,113],
                    [38,38,70,70]])
    '''
    print(boxes_o)
    print(boxes.numpy(),scores,clses)