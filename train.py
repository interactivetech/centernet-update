from data_gen import ShapeDataset
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from model import centernet
from utils import encode_hm_regr_and_wh_regr, decode_predictions
from loss import centerloss4
import torch
import cv2
import torch
from PIL import Image
from val import val
from efficient_centernet_model import EfficientCenternet, EfficientCenternet3, EfficientCenterDet

import torch.distributed as dist


def train(architecture='efd',
          num_classes=2,
          learn_rate=1e-4,
          epochs=10,
          train_loader=None,
          val_ds=None,
          val_loader=None,
          writer=None,
          multi_gpu=False,
          visualize_res=None,
          IMG_RESOLUTION=None,
          local_rank=None):
        if architecture == 'efd':
                model = EfficientCenterDet(num_classes=num_classes)

        
        if multi_gpu and local_rank is None:
                # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.DataParallel(model,device_ids=[0,1,2,3],output_device=[0])

                # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=[local_rank])
        elif multi_gpu and local_rank is not None:
                model = model.to(local_rank)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=[local_rank])


        EPOCHS = epochs
        LEARN_RATE = learn_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        # optimizer = torch.optim.SGD(model.parameters(),lr=LEARN_RATE,weight_decay=1e-4,momentum=0.9)
        DEVICE = None
        if torch.cuda.is_available() and local_rank is None:
                # DEVICE = torch.device('cuda:{}'.format(local_rank))
                DEVICE = torch.device('cuda:{}'.format(0))
        elif torch.cuda.is_available() and local_rank is not None:
                DEVICE = torch.device('cuda:{}'.format(local_rank))
        elif not torch.cuda.is_available():
                DEVICE='cpu'
        print(DEVICE)
        model.to(DEVICE)  # Move model to the device selected for training
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,300], gamma=0.1)
        losses = []
        mask_losses = []
        regr_losses = []
        min_confidences = []
        median_confidences = []
        max_confidences = []
        total_ind = 0
        for epoch in tqdm(range(EPOCHS)):
                for ind, (img,
                          hm, 
                          reg, 
                          wh,
                          reg_mask,
                          inds,
                          in_size,
                          out_size,
                          intermediate_size,
                          scale,
                          boxes_aug,
                          target,
                          idxs) in enumerate(tqdm(train_loader)):
                        # print(img.shape)
                        # print(in_size)
                        # print(out_size)
                        # print("idxs: ",idxs)
                        bboxes_gt = np.vstack([np.array(i['boxes']) for i in target])
                        # if DEVICE == 'cuda:0':
                        img = img.to(DEVICE).cuda(non_blocking=True)
                        hm = hm.to(DEVICE).cuda(non_blocking=True)
                        reg = reg.to(DEVICE).cuda(non_blocking=True)
                        wh = wh.to(DEVICE).cuda(non_blocking=True)
                        reg_mask = reg_mask.to(DEVICE).cuda(non_blocking=True)
                        inds = inds.to(DEVICE).cuda(non_blocking=True)
                        
                        optimizer.zero_grad()
                        # print("begin")
                        pred_hm, pred_regs = model(img)
                        # print("pred_hm: ",pred_hm.shape)


                        loss,mask_loss,regr_loss = centerloss(pred_hm,hm,pred_regs,reg,reg_mask,inds,wh)
                        # if local_rank is not None:
                        #         # torch.distributed.barrier()'s role is to block the process and ensure that each process runs all the code before this line of code before it can continue to execute, so that the average loss and average acc will not appear because of the process execution speed. inconsistent error
                        #         torch.distributed.barrier()
                        #         reduced_loss = reduce_mean(loss, 8)
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                                pred_hm = torch.sigmoid(pred_hm).detach()
                        p = pred_hm[pred_hm>0]
                        if len(p.size()) > 0 :
                                # print("Min confidence: {}, Median confidence: {}, Max Confidence: {}".format(p.min(), np.median(p), p.max()))
                                # print(p.min().item())
                                min_confidences.append(p.min().item())
                                median_confidences.append(p.median().item())
                                max_confidences.append(p.max().item())
                        else:
                                min_confidences.append(0.0)
                                median_confidences.append(0.0)
                                max_confidences.append(0.0)
                        losses.append(loss.item())
                        mask_losses.append(mask_loss.item())
                        regr_losses.append(regr_loss.item())
                        if local_rank is not None and local_rank ==0:
                                writer.add_scalar("Loss/train", loss.item(), total_ind)
                                writer.add_scalar("Mask Loss/train", mask_loss.item(), total_ind)
                                writer.add_scalar("Reg Loss/train", regr_loss.item(), total_ind)
                                writer.add_scalar("Max Conf Score/train", max_confidences[-1], total_ind)
                                lr = lr_scheduler.get_lr()[0]
                                # print(lr)
                                if writer is not None:
                                        writer.add_scalar("Learn Rate/train",lr, total_ind)


                        elif writer is not None:
                                writer.add_scalar("Loss/train", loss.item(), total_ind)
                                writer.add_scalar("Mask Loss/train", mask_loss.item(), total_ind)
                                writer.add_scalar("Reg Loss/train", regr_loss.item(), total_ind)
                                writer.add_scalar("Max Conf Score/train", max_confidences[-1], total_ind)
                                lr = lr_scheduler.get_lr()[0]
                                # print(lr)
                                if writer is not None:
                                        writer.add_scalar("Learn Rate/train",lr, total_ind)
                        total_ind+=1

                # if p.size > 0:
                #         print("Epoch {} - Min conf: {}, Median conf: {}, Max conf: {}".format(epoch,p.min(), np.median(p), p.max()))
                # else:
                #         print("Epoch {} - Min conf: {}, Median conf: {}, Max conf: {}".format(epoch,0,0,0))
                # print("Epoch {} - Loss: {}, Mask Loss: {}, Reg Loss: {}".format(epoch,loss.item(),mask_loss.item(),regr_loss.item()))
                # Save Example
                # if epoch > 0 and epoch%10==0:
                # print(img.shape)
                
                if epoch > 0 and epoch%100==0:
                        # Val
                        if local_rank is not None and local_rank==0:
                                # val(model,val_ds,val_loader, writer,epoch,visualize_res=visualize_res,IMG_RESOLUTION=IMG_RESOLUTION,device=DEVICE)
                                if multi_gpu:
                                        torch.save(model.module.state_dict(),'ddp_efficient_centernet_{}.pth'.format(epoch))
                                elif multi_gpu==False:
                                        torch.save(model.state_dict(),'efficient_centernet_{}.pth'.format(epoch))
                                model.train()
                        elif writer is not None:
                                # val(model,val_ds,val_loader, writer,epoch,visualize_res=visualize_res,IMG_RESOLUTION=IMG_RESOLUTION,device=None)
                                model.train()
                        # im0 = Image.fromarray(i0)
                        # im1 = Image.fromarray(i1)
                        # im0.save('hm_0.png')
                        # im1.save('hm_1.png')
                lr_scheduler.step()
        if writer is not None:
                writer.flush()
                writer.close()
        return model,losses,mask_losses,regr_losses, min_confidences, median_confidences, max_confidences