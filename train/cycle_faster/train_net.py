# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pprint
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model.cycle_faster_rcnn.resnet import resnet
from model.cycle_faster_rcnn.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import (
    adjust_learning_rate,
    clip_gradient,
    load_net,
    save_checkpoint,
    save_net,
    weights_normal_init,
)
from roi_da_data_layer.roibatchLoader import roibatchLoader
from roi_da_data_layer.roidb import combined_roidb
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from torch.utils.tensorboard import SummaryWriter

def infinite_data_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="cityscape",
        type=str,
    )
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="vgg16", type=str
    )
    parser.add_argument(
        "--pretrained_path",
        dest="pretrained_path",
        help="vgg16, res101",
        default="",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        help="number of iterations to save checkpoint",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="directory to save models",
        default=" ",
        type=str,
    )
    parser.add_argument(
        "--nw",
        dest="num_workers",
        help="number of worker to load data",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--bayes", dest="bayes", help="whether use bayes", action="store_true"
    )
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=1, type=int
    )


    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )

    # config optimization
    parser.add_argument(
        "--max_iter",
        dest="max_iter",
        help="max iteration for train",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--o", dest="optimizer", help="training optimizer", default="sgd", type=str
    )
    parser.add_argument(
        "--lr", dest="lr", help="starting learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--lr_decay_step",
        dest="lr_decay_step",
        help="step to do learning rate decay, unit is iter",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--lr_decay_gamma",
        dest="lr_decay_gamma",
        help="learning rate decay ratio",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--instance_weight_value",
        dest="instance_weight_value",
        help="instance_weight_value",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--lamda", dest="lamda", help="DA loss param", default=0.1, type=float
    )

    # set training session
    parser.add_argument(
        "--s", dest="session", help="training session", default=1, type=int
    )

    # resume trained model
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=False, type=bool
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="resume from which model",
        default="",
        type=str,
    )

    # setting display config
    parser.add_argument(
        "--disp_interval",
        dest="disp_interval",
        help="number of iterations to display",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--max_epochs",
        dest="max_epochs",
        help="max epoch for train",
        default=7,
        type=int,
    )
    parser.add_argument(
        "--start_epoch", dest="start_epoch", help="starting epoch", default=1, type=int
    )

    parser.add_argument(
        "--use_norm_da", dest="use_norm_da", help="use norm da or not", action="store_true"
    )

    parser.add_argument(
        "--use_img_da", dest="use_img_da", help="use img da or not", action="store_true"
    )

    parser.add_argument(
        "--use_sk_da", dest="use_sk_da", help="use sk da or not", action="store_true"
    )

    parser.add_argument(
        "--use_detect_da", dest="use_detect_da", help="use detect da or not", action="store_true"
    )

    parser.add_argument(
        "--chaos", dest="chaos", help="stat use net", action="store_true"
    )

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch * batch_size, train_size
            ).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = (
            rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        )

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == "__main__":

    args = parse_args()

    print("Called with args:")
    print(args)

    if args.dataset == "pascal_voc":
        print("loading our dataset...........")
        args.imdb_name = "voc_2007_train"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4,8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]
    elif args.dataset == "cityscape":
        print("loading our dataset...........")
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "cityscape_2007_test_t"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "bdd":
        print("loading our dataset...........")
        args.s_imdb_name = "citybdd7_train"
        args.t_imdb_name = "bdd_train"
        args.t_imdbtest_name = "bdd_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "clipart":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_2007_trainval"
        args.t_imdb_name = "clipart_train"
        args.t_imdbtest_name = "clipart_train"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]
    
    elif args.dataset == "LEVIR":
        print("loading our dataset...........")
        args.s_imdb_name = "LEVIR_trainval"
        args.t_imdb_name = "SSDD_train"
        args.t_imdbtest_name = "SSDD_train"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "SSDD":
        print("loading our dataset...........")
        args.s_imdb_name = "SSDD_train"
        args.t_imdb_name = "LEVIR_trainval"
        args.t_imdbtest_name = "LEVIR_trainval"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "water":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
        args.t_imdb_name = "water_train"
        args.t_imdbtest_name = "water_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]
    elif args.dataset == "sim10k":
        print("loading our dataset...........")
        args.s_imdb_name = "sim10k_2019_train"
        args.t_imdb_name = "cityscapes_car_2019_train"
        args.s_imdbtest_name = "sim10k_2019_val"
        args.t_imdbtest_name = "cityscapes_car_2019_val"
        args.set_cfgs = [
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "TRAIN.SCALES",
            "(800,)",
            "TRAIN.MAX_SIZE",
            "1600",
            "TEST.SCALES",
            "(800,)",
            "TEST.MAX_SIZE",
            "1600",
        ]

    args.cfg_file = (
        "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else "cfgs/{}.yml".format(args.net)
    )

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.s_imdb_name)
    s_train_size = len(s_roidb)  # add flipped         image_index*2

    t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name)
    t_train_size = len(t_roidb)  # add flipped         image_index*2

    print("source {:d} target {:d} roidb entries".format(len(s_roidb), len(t_roidb)))

    # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    output_dir = args.save_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    s_sampler_batch = sampler(s_train_size, args.batch_size)
    t_sampler_batch = sampler(t_train_size, args.batch_size)

    s_dataset = roibatchLoader(
        s_roidb,
        s_ratio_list,
        s_ratio_index,
        args.batch_size,
        s_imdb.num_classes,
        training=True,
    )

    s_dataloader = torch.utils.data.DataLoader(
        s_dataset,
        batch_size=args.batch_size,
        sampler=s_sampler_batch,
        num_workers=args.num_workers,
    )
    # s_dataloader = infinite_data_loader(s_dataloader)

    t_dataset = roibatchLoader(
        t_roidb,
        t_ratio_list,
        t_ratio_index,
        args.batch_size,
        t_imdb.num_classes,
        training=False,
    )

    t_dataloader = torch.utils.data.DataLoader(
        t_dataset,
        batch_size=args.batch_size,
        sampler=t_sampler_batch,
        num_workers=args.num_workers,
    )

    # t_dataloader = infinite_data_loader(t_dataloader)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_cls_lb = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    need_backprop = torch.FloatTensor(1)

    tgt_im_data = torch.FloatTensor(1)
    tgt_im_info = torch.FloatTensor(1)
    tgt_im_cls_lb = torch.FloatTensor(1)
    tgt_num_boxes = torch.LongTensor(1)
    tgt_gt_boxes = torch.FloatTensor(1)
    tgt_need_backprop = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_cls_lb = im_cls_lb.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        need_backprop = need_backprop.cuda()

        tgt_im_data = tgt_im_data.cuda()
        tgt_im_info = tgt_im_info.cuda()
        tgt_im_cls_lb = tgt_im_cls_lb.cuda()
        tgt_num_boxes = tgt_num_boxes.cuda()
        tgt_gt_boxes = tgt_gt_boxes.cuda()
        tgt_need_backprop = tgt_need_backprop.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_cls_lb = Variable(im_cls_lb)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    need_backprop = Variable(need_backprop)

    tgt_im_data = Variable(tgt_im_data)
    tgt_im_info = Variable(tgt_im_info)
    tgt_im_cls_lb = Variable(tgt_im_cls_lb)
    tgt_num_boxes = Variable(tgt_num_boxes)
    tgt_gt_boxes = Variable(tgt_gt_boxes)
    tgt_need_backprop = Variable(tgt_need_backprop)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            s_imdb.classes,
            pretrained=True,
            pretrained_path=args.pretrained_path,
            class_agnostic=args.class_agnostic,
        )
    elif args.net == "res101":
        fasterRCNN = resnet(
            s_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic
        )
    elif args.net == "res50":
        fasterRCNN = resnet(
            s_imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic
        )
    elif args.net == "res152":
        fasterRCNN = resnet(
            s_imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if "bias" in key:
                params += [
                    {
                        "params": [value],
                        "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        "weight_decay": cfg.TRAIN.BIAS_DECAY
                        and cfg.TRAIN.WEIGHT_DECAY
                        or 0,
                    }
                ]
            else:
                params += [
                    {
                        "params": [value],
                        "lr": lr,
                        "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                    }
                ]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)



    if args.resume:
        load_name = os.path.join(output_dir, args.model_name)
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint["session"]
        args.start_epoch = checkpoint["epoch"]
        fasterRCNN.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr = optimizer.param_groups[0]["lr"]
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (load_name))

    if args.cuda:
        fasterRCNN.cuda()

    data_iter = iter(s_dataloader)
    tgt_data_iter = iter(t_dataloader)
    loss_temp = 0

    iters_per_epoch = int(args.max_iter / args.batch_size)

    ##初始化tensorboard
    writer = SummaryWriter()

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
            #writer.add_scalar('LR/learning_rate', lr.item(), step + (epoch - 1) * iters_per_epoch)

        data_iter_s = iter(s_dataloader)
        data_iter_t = iter(t_dataloader)
        

        for step in range(iters_per_epoch):
            try:
                data = next(data_iter_s)
            except:
                data_iter_s = iter(s_dataloader)
                data = next(data_iter_s)
            try:
                tgt_data = next(data_iter_t)
            except:
                data_iter_t = iter(t_dataloader)
                tgt_data = next(data_iter_t)

            im_data.resize_(data[0].size()).copy_(data[0])  # change holder size
            im_info.resize_(data[1].size()).copy_(data[1])
            im_cls_lb.resize_(data[2].size()).copy_(data[2])
            gt_boxes.resize_(data[3].size()).copy_(data[3])
            num_boxes.resize_(data[4].size()).copy_(data[4])
            need_backprop.resize_(data[5].size()).copy_(data[5])

            tgt_im_data.resize_(tgt_data[0].size()).copy_(tgt_data[0])  # change holder size
            tgt_im_info.resize_(tgt_data[1].size()).copy_(tgt_data[1])
            tgt_im_cls_lb.resize_(data[2].size()).copy_(data[2])
            tgt_gt_boxes.resize_(tgt_data[3].size()).copy_(tgt_data[3])
            tgt_num_boxes.resize_(tgt_data[4].size()).copy_(tgt_data[4])
            tgt_need_backprop.resize_(tgt_data[5].size()).copy_(tgt_data[5])

            """   faster-rcnn loss + DA loss for source and   DA loss for target    """
            fasterRCNN.zero_grad()
            (
                rois,
                cls_prob,
                bbox_pred,
                tgt_rois, 
                tgt_cls_prob,
                tgt_bbox_pred,
                rpn_loss_cls,
                rpn_loss_bbox,
                tgt_rpn_loss_cls, 
                tgt_rpn_loss_bbox, 
                RCNN_loss_cls,
                RCNN_loss_bbox,
                tgt_RCNN_loss_cls, 
                tgt_RCNN_loss_bbox, 
                cst_bbox_loss,  
                tgt_cst_bbox_loss, 
                cst_cls_loss, 
                tgt_cst_cls_loss,
                source_skews, 
                target_skews, 
                source_kurtoses, 
                target_kurtoses, 
                img_cls, 
                fake_img_cls, 
                tgt_img_cls, 
                fake_tgt_img_cls, 
                source_norm_loss, 
                target_norm_loss
            ) = fasterRCNN(
                im_data,
                im_info,
                #im_cls_lb,
                gt_boxes,
                num_boxes,
                #need_backprop,
                tgt_im_data,
                tgt_im_info,
                tgt_gt_boxes,
                tgt_num_boxes,
                args.chaos,
                #tgt_need_backprop,
                #weight_value=args.instance_weight_value,
            )
            loss = (
                rpn_loss_cls.mean()
                + rpn_loss_bbox.mean()
                + tgt_rpn_loss_cls.mean()
                + tgt_rpn_loss_bbox.mean()
                + RCNN_loss_cls.mean()
                + RCNN_loss_bbox.mean()
                + tgt_RCNN_loss_cls.mean()
                + tgt_RCNN_loss_bbox.mean()
            )
            if args.use_img_da:
                loss += 1 * ( img_cls.mean()
                                + fake_img_cls.mean() 
                                + tgt_img_cls.mean()
                                + fake_tgt_img_cls.mean()
                                )
            
            if args.use_norm_da:
                loss += source_norm_loss.mean() 
                loss += target_norm_loss.mean() 
            
            if args.use_sk_da:
                loss += (source_skews.mean()
                         + target_skews.mean()
                         + source_kurtoses.mean()
                         + target_kurtoses.mean()
                         )
                loss += torch.abs(source_skews.mean() - target_skews.mean())
                loss += torch.abs(source_kurtoses.mean() - target_kurtoses.mean())
            
            if args.use_detect_da:
                loss += (cst_bbox_loss.mean()
                        + tgt_cst_bbox_loss.mean()
                        + cst_cls_loss.mean()
                        + tgt_cst_cls_loss.mean()
                        )
             
            loss_temp += loss.item()
            
            bayes_loss = 0
            loss += bayes_loss
            loss_temp += loss.item()

            real_step = step + (epoch - 1) * iters_per_epoch

            #writer.add_scalar('Detect_loss/rpn_cls', rpn_loss_cls.item(), real_step)
            #writer.add_scalar('Detect_loss/rpn_box', rpn_loss_box.item(), real_step)
            #writer.add_scalar('Detect_loss/rcnn_cls', RCNN_loss_cls.item(), real_step)
            #writer.add_scalar('Detect_loss/rcnn_box', RCNN_loss_bbox.item(), real_step)

            #writer.add_scalars('Da_loss/img_cls', {'source' : DA_img_loss_cls.item(),
            #                                      'target' : tgt_DA_img_loss_cls.item()}
            #                                      , real_step)

            #writer.add_scalars('Da_loss/ins_cls', {'source' : DA_ins_loss_cls.item(),
            #                                      'target' : tgt_DA_ins_loss_cls.item()}
            #                                      , real_step)
            #writer.add_scalars('Da_loss/cst_loss', {'source' : DA_cst_loss.item(),
            #                                        'target' : tgt_DA_cst_loss.item()}
            #                                        , real_step)
            
            if real_step % 100 == 0:
                writer.add_scalars('Da_stat/skews', {'source' : source_skews.item(),
                                                     'target' : target_skews.item()}
                                                     , real_step/100)

                writer.add_scalars('Da_stat/kurtoses', {'source' : source_kurtoses.item(),
                                                        'target' : target_kurtoses.item()}
                                                        , real_step/100)
            # backward
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(fasterRCNN, 10.0)  # 梯度裁剪
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.0)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval + 1

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_bbox.item()
                tgt_loss_rpn_cls = tgt_rpn_loss_cls.item()
                tgt_loss_rpn_box = tgt_rpn_loss_bbox.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                tgt_loss_rcnn_cls = tgt_RCNN_loss_cls.item()
                tgt_loss_rcnn_box = tgt_RCNN_loss_bbox.item()
                cst_bbox_loss = (cst_bbox_loss.item() + tgt_cst_bbox_loss.item())/2
                cst_cls_loss = (cst_cls_loss.item() + tgt_cst_cls_loss.item())/2
                source_norm_loss = source_norm_loss.item()
                target_norm_loss = target_norm_loss.item()

                #fg_cnt = torch.sum(rois_label.data.ne(0))
                #bg_cnt = rois_label.data.numel() - fg_cnt
                fg_cnt = 0
                bg_cnt = 0
         
                print(
                    "[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                    % (args.session, epoch, step, args.max_iter, loss_temp, lr)
                )
                print(
                    "\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start)
                )
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f,\n\t\t\ttgt_rpn_cls: %.4f,tgt_rpn_box: %.4f, tgt_rcnn_cls: %.4f, tgt_rcnn_box %.4f,\n\t\t\tcst_cls: %.4f, cst_bbox: %.4f, \n\t\t\tsource_skews: %.4f, target_skews: %.4f,\n\t\t\tsource_kurtoses: %.4f, target_kurtoses: %.4f, \n\t\t\tsource_cls: %.4f, fake_source_cls: %.4f, target_cls: %.4f, fake_target_cls: %.4f,\n\t\t\tsource_norm_cls: %.4f,target_norm_cls: %.4f"
                    % (
                        loss_rpn_cls,
                        loss_rpn_box,
                        loss_rcnn_cls,
                        loss_rcnn_box,
                        tgt_loss_rpn_cls,
                        tgt_loss_rpn_box,
                        tgt_loss_rcnn_cls,
                        tgt_loss_rcnn_box,
                        cst_cls_loss, 
                        cst_bbox_loss, 
                        source_skews, 
                        target_skews, 
                        source_kurtoses, 
                        target_kurtoses,
                        img_cls, 
                        fake_img_cls, 
                        tgt_img_cls, 
                        fake_tgt_img_cls, 
                        source_norm_loss, 
                        target_norm_loss
                    )
                )

                loss_temp = 0
                start = time.time()
                
                #writer.add_images('im_data', im_data, real_step)
                #writer.add_images('tgt_im_data', tgt_im_data, real_step)
                #writer.add_images('base_feat', base_feat[:,:3], real_step)
                #writer.add_images('tgt_base_feat', tgt_base_feat[:,:3], real_step)
                
        if epoch % args.checkpoint_interval == 0 or epoch == args.max_epochs:
            save_name = os.path.join(
                output_dir, "{}.pth".format(args.dataset + "_" + str(epoch)),
            )
            save_checkpoint(
                {
                    "session": args.session,
                    "iter": step + 1,
                    "model": fasterRCNN.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pooling_mode": cfg.POOLING_MODE,
                    "class_agnostic": args.class_agnostic,
                },
                save_name,
            )
            print("save model: {}".format(save_name))
