import torch
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from loss import centerloss
from train import train
from utils import decode_predictions, rescale_boxes
import cv2
from val import val, torch_to_numpy
import matplotlib.pyplot as plt
import random
import os

import argparse
import datetime
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)

def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()

def main(args):

    torch.cuda.set_device ( args.local_rank ) # use set_device and cuda to specify the desired GPU
    torch.distributed.init_process_group(
    'nccl',
        init_method='env://',
        world_size=4,
        timeout=datetime.timedelta(seconds=18000),
        rank=args.local_rank,
    )
    IMG_RESOLUTION=256 
    MODEL_SCALE=2

    # ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/',
    #                      '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/coco_shapes.json',
    #                      MODEL_SCALE=MODEL_SCALE,
    #                      transform=train_transform_norm,
    #                      IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/',
    #                      '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/coco_shapes.json',
    #                      MODEL_SCALE=MODEL_SCALE,
    #                      transform=validation_transform_norm,
    #                      IMG_RESOLUTION=IMG_RESOLUTION)
    # # Mini COCO Dataset
    ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/train2017',
    '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_minitrain2017.json',
    transform=train_transform_norm,
    MODEL_SCALE=MODEL_SCALE,
    IMG_RESOLUTION=IMG_RESOLUTION)
    val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/val2017',
    '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_val2017.json',
    transform=validation_transform_norm,
    MODEL_SCALE=MODEL_SCALE,
    IMG_RESOLUTION=IMG_RESOLUTION)

    # DATASET_DIR = '/run/determined/workdir/coco_dataset'
    # IMG_RESOLUTION=256 
    # MODEL_SCALE=2
    # # Mini COCO Dataset
    # ds = COCODetectionDataset(os.path.join(DATASET_DIR,'train2017'),
    # os.path.join(DATASET_DIR,'annotations/instances_minitrain2017.json'),
    # transform=train_transform_norm,
    # MODEL_SCALE=MODEL_SCALE,
    # IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset(os.path.join(DATASET_DIR,'val2017'),
    # os.path.join(DATASET_DIR,'annotations/instances_val2017.json'),
    # transform=train_transform_norm,
    # MODEL_SCALE=MODEL_SCALE,
    # IMG_RESOLUTION=IMG_RESOLUTION)
    

    BATCH_SIZE = 12
    # BATCH_SIZE = 8

    samp = torch.utils.data.distributed.DistributedSampler(ds,shuffle=True)
    # t_sampler = torch.utils.data.distributed.DistributedSampler(val_ds,shuffle=False)
    # generator = torch.Generator()
    # generator.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=BATCH_SIZE,
                                            num_workers=8,
                                            pin_memory=True,
                                            sampler=samp,
                                            collate_fn = coco_detection_collate_fn,
                                            generator=None)
    print("Len of Train Loader: ",len(train_loader))
    # train_loader = torch.utils.data.DataLoader(ds,
    #                                         batch_size=BATCH_SIZE,
    #                                         shuffle=True,
    #                                         num_workers=8,
    #                                         pin_memory=True,
    #                                         collate_fn = coco_detection_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn)
    print("Len of Val Loader: ",len(val_loader))

    # LR = 1e-3
    LR = 1e-2
    # LR = 2.5e-4*BATCH_SIZE
    from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(comment='mv2')
    # writer = SummaryWriter(comment='_mini_coco_emv2')
    print("args.local_rank: ",args.local_rank)
    writer=None
    if args.local_rank == 0:
        # writer = SummaryWriter(comment='_shape_mv2')
        writer = SummaryWriter(comment='_efd')


    multi_gpu=True
    # visualize_res=IMG_RESOLUTION//4
    visualize_res=IMG_RESOLUTION//2
    EPOCHS = 401
    # model, losses, mask_losses, regr_losses, min_confidences, median_confidences, max_confidences = train('mv2',
    #                                                                                                         ds.num_classes,
    #                                                                                                         learn_rate=LR,
    #                                                                                                         epochs=300,
    #                                                                                                         train_loader=train_loader,
    #                                                                                                         val_ds=val_ds,
    #                                                                                                         val_loader=val_loader,
    #                                                                                                         writer=writer,
    #                                                                                                         multi_gpu=multi_gpu,
    #                                                                                                         visualize_res=visualize_res,
    #                                                                                                         IMG_RESOLUTION=IMG_RESOLUTION)
    model,losses,mask_losses,regr_losses,wh_regr_losses, min_confidences, median_confidences, max_confidences = train('efd',
                                                                                                            ds.num_classes,
                                                                                                            learn_rate=LR,
                                                                                                            epochs=EPOCHS,
                                                                                                            train_loader=train_loader,
                                                                                                            val_ds=val_ds,
                                                                                                            val_loader=val_loader,
                                                                                                            writer=writer,
                                                                                                            multi_gpu=multi_gpu,
                                                                                                            visualize_res=visualize_res,
                                                                                                            IMG_RESOLUTION=IMG_RESOLUTION,
                                                                                                            local_rank=args.local_rank,
                                                                                                            MODEL_SCALE=MODEL_SCALE)
    if multi_gpu:
        torch.save(model.module.state_dict(),'ddp_efficient_centernet_{}_coco.pth'.format(EPOCHS))
    else:
        torch.save(model.state_dict(),'efficient_centernet_{}.pth'.format(EPOCHS))
    plt.plot(range(len(losses)),losses )
    plt.plot(range(len(losses)),mask_losses)
    plt.plot(range(len(losses)),regr_losses)
    plt.plot(range(len(losses)),wh_regr_losses)
    plt.title("centernet (mobilenetv3 backbone) training")
    plt.legend(['loss','mask loss','regression loss'])
    # plt.show()
    plt.savefig("loss.png")
    plt.clf()
        


    plt.plot(range(len(min_confidences)),min_confidences )
    plt.plot(range(len(median_confidences)),median_confidences)
    plt.plot(range(len(max_confidences)),max_confidences)
    plt.title("confidence scores for centernet (mv3 backbone) during training")
    plt.legend(['min_confidences','median_confidences','max_confidences'])
    # plt.show()
    plt.savefig("conf.png")
    plt.clf()
    model.eval()
    # model.cpu()

    # eval
    # val(model,val_ds,val_loader,writer,epoch)
    if args.local_rank == 0:
        with torch.no_grad():
            for img,hm, regr, wh_regr,wh,inds_mask,inds,regr_,in_size, out_size, intermediate_size, scale, boxes_aug, target, idxs in tqdm(val_loader):
                    break

            pred_hm, pred_regr,pred_wh_regr = model(img)# (4,1,128,128), (4,2,128,128)
            pred_hm = torch.sigmoid(pred_hm)
            # bboxes,scores,classes = pred2box_multiclass(pred_hm[0].cpu().data.numpy(),
            #                                                         pred_regs[0].cpu().detach().numpy(),128,1,thresh=0.0)
            for ind in range(pred_hm.shape[0]):
                if torch.cuda.is_available():
                    # bboxes,scores,classes = decode_predictions(pred_hm,pred_regs,visualize_res,1,thresh=0.25)
                    boxes,scores,clses = decode_predictions(pred_hm[ind].unsqueeze(0),
                                                            pred_regr[ind].unsqueeze(0),
                                                            pred_wh_regr[ind].unsqueeze(0),
                                                            MODEL_SCALE=MODEL_SCALE,
                                                            K=100,
                                                            nms_thresh=0.5)
                else:
                    boxes,scores,clses = decode_predictions(pred_hm[ind].unsqueeze(0),
                                                            pred_regr[ind].unsqueeze(0),
                                                            pred_wh_regr[ind].unsqueeze(0),
                                                            MODEL_SCALE=MODEL_SCALE,
                                                            K=100,
                                                            nms_thresh=0.5)
            # bboxes,scores,classes =  filter_and_nms(bboxes,scores,classes,nms_threshold=0.45,n_top_scores=100)
            boxes,scores,clses = torch_to_numpy(boxes,scores,clses)
                                # print("boxes: ",boxes[0])
            boxes = rescale_boxes( boxes, out_size[ind].numpy(), intermediate_size[ind].numpy(), scale[ind].numpy())
            print(boxes)

#     for i in range(hm.shape[1]):
#         if torch.cuda.is_available():
#             hm_gt = hm[0].cpu().data.numpy()[i]
#             hm_pred = pred_hm[0].cpu().data.numpy()[i]
#         else:
#             hm_gt = hm[0].data.numpy()[i]
#             hm_pred = pred_hm[0].data.numpy()[i]
#         hm_pred = np.dstack([hm_pred*255]*3).astype(np.uint8)
#         for b,c in zip(bboxes,classes):
#             if c == 0:
#                 x,y,x2,y2 = [int(k) for k in b]
#                 # print(x,y)
#                 cv2.rectangle(hm_pred,(x,y),(x2,y2),(255,0,0),1)
#             if c == 1:
#                 x,y,x2,y2 = [int(k) for k in b]
#                 # print(x,y)
#                 cv2.rectangle(hm_pred,(x,y),(x2,y2),(0,255,0),1)
            
#         plt.imshow(hm_gt,cmap='gray')
#         plt.title("GT centerpoints of Class {}".format(i))
#         plt.imshow(hm_gt,cmap='gray')
#         plt.savefig("gt_hm.png")
#         plt.clf()

#         plt.title("prediction centerpoints for Class {} from model".format(i))
#         plt.imshow(hm_pred)
#         plt.savefig("hm_preds.png")
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)