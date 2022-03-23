import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.detect_vae.DA import _ImageDA, _InstanceDA
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
)
from torch.autograd import Variable

from model.detect_vae.GMM import Random_detect_map, _to_detect_map, four2three, Encoder, Decoder
from model.detect_vae.JS import js_map_loss, kl_map_loss


#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, in_channel=4096, bayes=0):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.bayes = bayes
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
   
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )
        # self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA(in_channel)
        self.consistency_loss = torch.nn.MSELoss(size_average=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        #self.Get_detect_map = _to_detect_map(self.n_classes - 1)
        self.Random_detect_map = Random_detect_map(self.n_classes - 1)
        #self.four2three = four2three(self.n_classes - 1)
        self.Encoder = Encoder(self.n_classes - 1)
        self.Decoder = Decoder(self.n_classes - 1)

    def forward(
        self,
        im_data,
        im_info,
        gt_boxes,
        num_boxes,
        tgt_im_data,
        tgt_im_info,
        tgt_gt_boxes,
        tgt_num_boxes,
    ):

        need_backprop = torch.Tensor([1]).cuda()
        tgt_need_backprop = torch.Tensor([0]).cuda()

        batch_size = im_data.size(0)
        im_info = im_info.data  # (size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop = need_backprop.data
        
        # feed image data to base model to obtain base feature map
        #source_detect_map = self.Get_detect_map(im_data)
        source_detect_map, source_other_feat = self.Encoder(im_data, domain=0)
        source_rand_detect_map, rand_gt_boxes, rand_num_boxes = self.Random_detect_map(im_data.shape)

        #source_im_data = torch.cat([im_data, source_detect_map], dim=1)
        #source_rand_data = torch.cat([im_data, source_rand_detect_map], dim=1)
        
        source_im_data = self.Decoder(source_detect_map, source_other_feat)
        source_rand_data = self.Decoder(source_rand_detect_map, source_other_feat)

        base_feat = self.RCNN_base(source_im_data)
        rand_base_feat = self.RCNN_base(source_rand_data)

        # feed base feature map tp RPN to obtain rois
        if self.training:
            self.RCNN_rpn.train()

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes
        )
        
        rand_rois, rand_rpn_loss_cls, rand_rpn_loss_bbox = self.RCNN_rpn(
            rand_base_feat, im_info, rand_gt_boxes, rand_num_boxes
        )

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2))
            )

            rand_roi_data = self.RCNN_proposal_target(rand_rois, rand_gt_boxes, rand_num_boxes)
            rand_rois, rand_rois_label, rand_rois_target, rand_rois_inside_ws, rand_rois_outside_ws = rand_roi_data

            rand_rois_label = Variable(rand_rois_label.view(-1).long())
            rand_rois_target = Variable(rand_rois_target.view(-1, rand_rois_target.size(2)))
            rand_rois_inside_ws = Variable(rand_rois_inside_ws.view(-1, rand_rois_inside_ws.size(2)))
            rand_rois_outside_ws = Variable(
                rand_rois_outside_ws.view(-1, rand_rois_outside_ws.size(2))
            )

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

            rand_rois_label = None
            rand_rois_target = None
            rand_rois_inside_ws = None
            rand_rois_outside_ws = None
            rand_rpn_loss_cls = 0
            rand_rpn_loss_bbox = 0

        rois = Variable(rois)
        rand_rois = Variable(rand_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "align":
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            rand_pooled_feat = self.RCNN_roi_align(rand_base_feat, rand_rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            rand_pooled_feat = self.RCNN_roi_pool(rand_base_feat, rand_rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        rand_pooled_feat = self._head_to_tail(rand_pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        rand_bbox_pred = self.RCNN_bbox_pred(rand_pooled_feat)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4
            )
            bbox_pred_select = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4
                ),
            )
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        rand_cls_score = self.RCNN_cls_score(rand_pooled_feat)
        rand_cls_prob = F.softmax(rand_cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        rand_RCNN_loss_cls = 0
        rand_RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            rand_RCNN_loss_cls = F.cross_entropy(rand_cls_score, rand_rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
            )
            rand_RCNN_loss_bbox = _smooth_l1_loss(
                rand_bbox_pred, rand_rois_target, rand_rois_inside_ws, rand_rois_outside_ws
            )

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        rand_cls_prob = rand_cls_prob.view(batch_size, rand_rois.size(1), -1)
        rand_bbox_pred = rand_bbox_pred.view(batch_size, rand_rois.size(1), -1)

        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = (
            tgt_im_info.data
        )  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        #tgt_detect_map = self.Get_detect_map(tgt_im_data)
        tgt_detect_map, tgt_other_feat = self.Encoder(tgt_im_data, domain=1)
        tgt_rand_detect_map, tgt_rand_gt_boxes, tgt_rand_num_boxes = self.Random_detect_map(tgt_im_data.shape)
        tgt_rand2_detect_map, tgt_rand2_gt_boxes, tgt_rand2_num_boxes = self.Random_detect_map(tgt_im_data.shape)
        #target_im_data = torch.cat([tgt_im_data, tgt_detect_map], dim=1)
        #target_rand_data = torch.cat([tgt_im_data, tgt_rand_detect_map], dim=1)

        target_im_data = self.Decoder(tgt_detect_map, tgt_other_feat)
        target_rand_data = self.Decoder(tgt_rand_detect_map, tgt_other_feat)
        target_rand2_data = self.Decoder(tgt_rand2_detect_map, tgt_other_feat)

        # feed image data to base model to obtain base feature map

        tgt_rand_base_feat = self.RCNN_base(target_rand_data)
        tgt_rand2_base_feat = self.RCNN_base(target_rand2_data)
        # use target random detect map for training 
        if self.training:
             self.RCNN_rpn.train()
        tgt_rand_rois, tgt_rand_rpn_loss_cls, tgt_rand_rpn_loss_bbox = self.RCNN_rpn(
            tgt_rand_base_feat, tgt_im_info, tgt_rand_gt_boxes, tgt_rand_num_boxes
        )
        tgt_rand2_rois, tgt_rand2_rpn_loss_cls, tgt_rand2_rpn_loss_bbox = self.RCNN_rpn(
            tgt_rand2_base_feat, tgt_im_info, tgt_rand2_gt_boxes, tgt_rand2_num_boxes
        )

        if self.training:
            tgt_rand_roi_data = self.RCNN_proposal_target(tgt_rand_rois, tgt_rand_gt_boxes, tgt_rand_num_boxes)
            tgt_rand_rois, tgt_rand_rois_label, tgt_rand_rois_target, tgt_rand_rois_inside_ws, tgt_rand_rois_outside_ws = tgt_rand_roi_data

            tgt_rand_rois_label = Variable(tgt_rand_rois_label.view(-1).long())
            tgt_rand_rois_target = Variable(tgt_rand_rois_target.view(-1, tgt_rand_rois_target.size(2)))
            tgt_rand_rois_inside_ws = Variable(tgt_rand_rois_inside_ws.view(-1, tgt_rand_rois_inside_ws.size(2)))
            tgt_rand_rois_outside_ws = Variable(
                tgt_rand_rois_outside_ws.view(-1, tgt_rand_rois_outside_ws.size(2))
            )

            tgt_rand2_roi_data = self.RCNN_proposal_target(tgt_rand2_rois, tgt_rand2_gt_boxes, tgt_rand2_num_boxes)
            tgt_rand2_rois, tgt_rand2_rois_label, tgt_rand2_rois_target, tgt_rand2_rois_inside_ws, tgt_rand2_rois_outside_ws = tgt_rand2_roi_data

            tgt_rand2_rois_label = Variable(tgt_rand2_rois_label.view(-1).long())
            tgt_rand2_rois_target = Variable(tgt_rand2_rois_target.view(-1, tgt_rand2_rois_target.size(2)))
            tgt_rand2_rois_inside_ws = Variable(tgt_rand2_rois_inside_ws.view(-1, tgt_rand2_rois_inside_ws.size(2)))
            tgt_rand2_rois_outside_ws = Variable(
                tgt_rand2_rois_outside_ws.view(-1, tgt_rand2_rois_outside_ws.size(2))
            )

        else:
            tgt_rand_rois_label = None
            tgt_rand_rois_target = None
            tgt_rand_rois_inside_ws = None
            tgt_rand_rois_outside_ws = None
            tgt_rand_rpn_loss_cls = 0
            tgt_rand_rpn_loss_bbox = 0

            tgt_rand2_rois_label = None
            tgt_rand2_rois_target = None
            tgt_rand2_rois_inside_ws = None
            tgt_rand2_rois_outside_ws = None
            tgt_rand2_rpn_loss_cls = 0
            tgt_rand2_rpn_loss_bbox = 0

        tgt_rand_rois = Variable(tgt_rand_rois)
        tgt_rand2_rois = Variable(tgt_rand2_rois)
        
        if cfg.POOLING_MODE == "align":
            tgt_rand_pooled_feat = self.RCNN_roi_align(tgt_rand_base_feat, tgt_rand_rois.view(-1, 5))
            tgt_rand2_pooled_feat = self.RCNN_roi_align(tgt_rand2_base_feat, tgt_rand2_rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            tgt_rand_pooled_feat = self.RCNN_roi_pool(tgt_rand_base_feat, tgt_rand_rois.view(-1, 5))
            tgt_rand2_pooled_feat = self.RCNN_roi_pool(tgt_rand2_base_feat, tgt_rand2_rois.view(-1, 5))

        tgt_rand_pooled_feat = self._head_to_tail(tgt_rand_pooled_feat)
        tgt_rand_bbox_pred = self.RCNN_bbox_pred(tgt_rand_pooled_feat)
        
        tgt_rand2_pooled_feat = self._head_to_tail(tgt_rand2_pooled_feat)
        tgt_rand2_bbox_pred = self.RCNN_bbox_pred(tgt_rand2_pooled_feat)
        
        tgt_rand_cls_score = self.RCNN_cls_score(tgt_rand_pooled_feat)
        tgt_rand_cls_prob = F.softmax(tgt_rand_cls_score, 1)

        tgt_rand2_cls_score = self.RCNN_cls_score(tgt_rand2_pooled_feat)
        tgt_rand2_cls_prob = F.softmax(tgt_rand2_cls_score, 1)

        tgt_rand_RCNN_loss_cls = 0
        tgt_rand_RCNN_loss_bbox = 0
        tgt_rand2_RCNN_loss_cls = 0
        tgt_rand2_RCNN_loss_bbox = 0
        

        if self.training:
            # classification loss
            tgt_rand_RCNN_loss_cls = F.cross_entropy(tgt_rand_cls_score, tgt_rand_rois_label)
            tgt_rand2_RCNN_loss_cls = F.cross_entropy(tgt_rand2_cls_score, tgt_rand2_rois_label)

            # bounding box regression L1 loss
            tgt_rand_RCNN_loss_bbox = _smooth_l1_loss(
                tgt_rand_bbox_pred, tgt_rand_rois_target, tgt_rand_rois_inside_ws, tgt_rand_rois_outside_ws
            )
            tgt_rand2_RCNN_loss_bbox = _smooth_l1_loss(
                tgt_rand2_bbox_pred, tgt_rand2_rois_target, tgt_rand2_rois_inside_ws, tgt_rand2_rois_outside_ws
            )

        tgt_rand_cls_prob = tgt_rand_cls_prob.view(batch_size, tgt_rand_rois.size(1), -1)
        tgt_rand_bbox_pred = tgt_rand_bbox_pred.view(batch_size, tgt_rand_rois.size(1), -1)

        tgt_rand2_cls_prob = tgt_rand2_cls_prob.view(batch_size, tgt_rand2_rois.size(1), -1)
        tgt_rand2_bbox_pred = tgt_rand2_bbox_pred.view(batch_size, tgt_rand2_rois.size(1), -1)

        #construct loss 
        source_cyc_loss = nn.L1Loss(reduction='mean')(im_data, source_im_data)
        target_cyc_loss = nn.L1Loss(reduction='mean')(tgt_im_data, target_im_data)
        ######## map js loss 
        sorce_map_loss = js_map_loss(source_detect_map, gt_boxes, num_boxes)

        return (
            rois,
            cls_prob,
            bbox_pred,
            rpn_loss_cls,
            rpn_loss_bbox,
            RCNN_loss_cls,
            RCNN_loss_bbox,
            rand_rpn_loss_cls,
            rand_rpn_loss_bbox, 
            rand_RCNN_loss_cls, 
            rand_RCNN_loss_bbox,
            tgt_rand_rpn_loss_cls,
            tgt_rand_rpn_loss_bbox,
            tgt_rand_RCNN_loss_cls,
            tgt_rand_RCNN_loss_bbox,
            tgt_rand2_rpn_loss_cls,
            tgt_rand2_rpn_loss_bbox,
            tgt_rand2_RCNN_loss_cls,
            tgt_rand2_RCNN_loss_bbox,
            source_cyc_loss, 
            target_cyc_loss, 
            sorce_map_loss, 
            im_data, 
            source_im_data, 
            source_rand_data, 
            tgt_im_data, 
            target_im_data, 
            target_rand_data, 
            target_rand2_data, 
            rand_gt_boxes, 
            tgt_rand_gt_boxes, 
            tgt_rand2_gt_boxes, 
        )

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_imageDA.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_imageDA.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.dc_ip1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.dc_ip2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.clssifer, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck.conv1, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck.conv2, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck.conv3, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck2.conv1, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck2.conv2, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck2.conv3, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck3.conv1, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck3.conv2, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Encoder.bottleneck3.conv3, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck.conv1, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck.conv2, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck.conv3, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck2.conv1, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck2.conv2, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck2.conv3, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck3.conv1, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck3.conv2, 0, 0.05, cfg.TRAIN.TRUNCATED)
        normal_init(self.Decoder.bottleneck3.conv3, 0, 0.05, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
