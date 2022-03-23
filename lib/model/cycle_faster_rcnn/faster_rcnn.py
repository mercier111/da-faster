import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.cycle_faster_rcnn.DA import _ImageDA, _InstanceDA
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

from model.cycle_faster_rcnn.adain import Adain, Adain_with_da, Adain_with_chaos, Adain_with_attention


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
        self.tgt_RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
    
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )
        # self.RCNN_roi_crop = _RoICrop()

        self.source_D = _ImageDA(self.dout_base_model)
        self.target_D = _ImageDA(self.dout_base_model)
        self.adain = Adain_with_da(self.dout_base_model)
        self.adain_chaos = Adain_with_chaos(self.dout_base_model) 
        #self.adain_attention = Adain_with_attention(self.dout_base_model) 
        self.source_instance = _InstanceDA(in_channel)
        self.target_instance = _InstanceDA(in_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(
        self,
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
        chaos=0, 
        attention = 0
        #tgt_need_backprop,
    ):

        need_backprop = torch.Tensor([1]).cuda()
        tgt_need_backprop = torch.Tensor([0]).cuda()

        batch_size = im_data.size(0)
        im_info = im_info.data  # (size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop = need_backprop.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = (
            tgt_im_info.data
        )  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed image data to base model to obtain base feature map
        tgt_base_feat = self.RCNN_base(tgt_im_data)
        if chaos:
            (   fake_base_feat, fake_tgt_base_feat, 
                source_skews, target_skews, source_kurtoses, target_kurtoses, 
                source_norm_loss, target_norm_loss
             ) = self.adain_chaos(base_feat, tgt_base_feat)
        #elif attention:
        #    (   fake_base_feat, fake_tgt_base_feat, 
        #        source_skews, target_skews, source_kurtoses, target_kurtoses, 
        #        source_norm_loss, target_norm_loss
        #     ) = self.adain_attention(base_feat, tgt_base_feat)
        else:
            (   fake_base_feat, fake_tgt_base_feat, 
                source_skews, target_skews, source_kurtoses, target_kurtoses, 
                source_norm_loss, target_norm_loss
             ) = self.adain(base_feat, tgt_base_feat)

        # feed base feature map tp RPN to obtain rois
        if self.training:
            self.RCNN_rpn.train()
            self.tgt_RCNN_rpn.train()

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes
        )

        fake_tgt_rois, fake_tgt_rpn_loss_cls, fake_tgt_rpn_loss_bbox = self.tgt_RCNN_rpn(
            fake_tgt_base_feat, im_info, gt_boxes, num_boxes
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

            fake_tgt_roi_data = self.RCNN_proposal_target(fake_tgt_rois, gt_boxes, num_boxes)
            fake_tgt_rois, fake_tgt_rois_label, fake_tgt_rois_target, fake_tgt_rois_inside_ws, fake_tgt_rois_outside_ws = fake_tgt_roi_data

            fake_tgt_rois_label = Variable(fake_tgt_rois_label.view(-1).long())
            fake_tgt_rois_target = Variable(fake_tgt_rois_target.view(-1, fake_tgt_rois_target.size(2)))
            fake_tgt_rois_inside_ws = Variable(fake_tgt_rois_inside_ws.view(-1, fake_tgt_rois_inside_ws.size(2)))
            fake_tgt_rois_outside_ws = Variable(
                fake_tgt_rois_outside_ws.view(-1, fake_tgt_rois_outside_ws.size(2))
            )

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

            fake_tgt_rois_label = None
            fake_tgt_rois_target = None
            fake_tgt_rois_inside_ws = None
            fake_tgt_rois_outside_ws = None
            fake_tgt_rpn_loss_cls = 0
            fake_tgt_rpn_loss_bbox = 0

        rois = Variable(rois)
        fake_tgt_rois = Variable(fake_tgt_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "align":
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            fake_tgt_pooled_feat = self.RCNN_roi_align(fake_tgt_base_feat, fake_tgt_rois.view(-1, 5))

        elif cfg.POOLING_MODE == "pool":
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            fake_tgt_pooled_feat = self.RCNN_roi_pool(fake_tgt_base_feat, fake_tgt_rois.view(-1, 5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        fake_tgt_pooled_feat = self._head_to_tail(fake_tgt_pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        fake_tgt_bbox_pred = self.tgt_RCNN_bbox_pred(fake_tgt_pooled_feat)

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

            fake_tgt_bbox_pred_view = fake_tgt_bbox_pred.view(
                fake_tgt_bbox_pred.size(0), int(fake_tgt_bbox_pred.size(1) / 4), 4
            )
            fake_tgt_bbox_pred_select = torch.gather(
                fake_tgt_bbox_pred_view,
                1,
                fake_tgt_rois_label.view(fake_tgt_rois_label.size(0), 1, 1).expand(
                    fake_tgt_rois_label.size(0), 1, 4
                ),
            )
            fake_tgt_bbox_pred = fake_tgt_bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        fake_tgt_cls_score = self.tgt_RCNN_cls_score(fake_tgt_pooled_feat)
        fake_tgt_cls_prob = F.softmax(fake_tgt_cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        fake_tgt_RCNN_loss_cls = 0
        fake_tgt_RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            fake_tgt_RCNN_loss_cls = F.cross_entropy(fake_tgt_cls_score, fake_tgt_rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
            )

            fake_tgt_RCNN_loss_bbox = _smooth_l1_loss(
                fake_tgt_bbox_pred, fake_tgt_rois_target, fake_tgt_rois_inside_ws, fake_tgt_rois_outside_ws
            )

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        fake_tgt_cls_prob = fake_tgt_cls_prob.view(batch_size, fake_tgt_rois.size(1), -1)
        fake_tgt_bbox_pred = fake_tgt_bbox_pred.view(batch_size, fake_tgt_rois.size(1), -1)

        """ =================== eval phraze =========================="""

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(
            tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes
        )
        self.tgt_RCNN_rpn.eval()
        fake_rois, fake_rpn_loss_cls, fake_rpn_loss_bbox = self.tgt_RCNN_rpn(
            fake_base_feat, tgt_im_info, tgt_gt_boxes, tgt_gt_boxes
        )

        # if it is training phrase, then use ground trubut bboxes for refining

        tgt_rois_label = None
        tgt_rois_target = None
        tgt_rois_inside_ws = None
        tgt_rois_outside_ws = None
        tgt_rpn_loss_cls = 0
        tgt_rpn_loss_bbox = 0

        fake_rois_label = None
        fake_rois_target = None
        fake_rois_inside_ws = None
        fake_rois_outside_ws = None
        fake_rpn_loss_cls = 0
        fake_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        fake_rois = Variable(fake_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "crop":
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            tgt_grid_xy = _affine_grid_gen(
                tgt_rois.view(-1, 5), tgt_base_feat.size()[2:], self.grid_size
            )
            tgt_grid_yx = torch.stack(
                [tgt_grid_xy.data[:, :, :, 1], tgt_grid_xy.data[:, :, :, 0]], 3
            ).contiguous()
            tgt_pooled_feat = self.RCNN_roi_crop(
                tgt_base_feat, Variable(tgt_grid_yx).detach()
            )
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                tgt_pooled_feat = F.max_pool2d(tgt_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == "align":
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
            fake_pooled_feat = self.RCNN_roi_align(fake_base_feat, fake_rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))
            fake_pooled_feat = self.RCNN_roi_pool(fake_base_feat, fake_rois.view(-1, 5))

        # feed pooled features to top model
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)
        fake_pooled_feat = self._head_to_tail(fake_pooled_feat)
        
        if tgt_pooled_feat.shape[0] > pooled_feat.shape[0]:
            tgt_pooled_feat = tgt_pooled_feat[: pooled_feat.shape[0]]
            fake_pooled_feat = fake_pooled_feat[: pooled_feat.shape[0]]

        tgt_bbox_pred = self.tgt_RCNN_bbox_pred(tgt_pooled_feat)
        fake_bbox_pred = self.RCNN_bbox_pred(fake_pooled_feat)
        
        tgt_cls_score = self.tgt_RCNN_cls_score(tgt_pooled_feat)
        tgt_cls_prob = F.softmax(tgt_cls_score, 1)

        fake_cls_score = self.RCNN_cls_score(fake_pooled_feat)
        fake_cls_prob = F.softmax(fake_cls_score, 1)

        tgt_cls_prob = tgt_cls_prob.view(batch_size, rois.size(1), -1)
        tgt_bbox_pred = tgt_bbox_pred.view(batch_size, rois.size(1), -1)

        fake_cls_prob = fake_cls_prob.view(batch_size, rois.size(1), -1)
        fake_bbox_pred = fake_bbox_pred.view(batch_size, rois.size(1), -1)
        #########consistant loss 
        
        smooth_l1 = nn.SmoothL1Loss()
        #cst_bbox_loss = smooth_l1(bbox_pred, fake_tgt_bbox_pred)
        tgt_cst_bbox_loss = smooth_l1(tgt_bbox_pred, fake_bbox_pred)
        cst_bbox_loss = torch.Tensor([0]).cuda()

        #cst_cls_loss = smooth_l1(cls_prob, fake_tgt_cls_prob)
        tgt_cst_cls_loss = smooth_l1(tgt_cls_prob, fake_cls_prob)
        cst_cls_loss = torch.Tensor([0]).cuda()
        
        base_score, base_label = self.source_D(base_feat, need_backprop)
        base_prob = F.log_softmax(base_score, dim=1)
        img_cls = F.nll_loss(base_prob, base_label)

        fake_base_score, fake_base_label = self.source_D(fake_base_feat, tgt_need_backprop)
        fake_base_prob = F.log_softmax(fake_base_score, dim=1)
        fake_img_cls = F.nll_loss(fake_base_prob, fake_base_label)

        tgt_base_score, tgt_base_label = self.target_D(tgt_base_feat, tgt_need_backprop)
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        tgt_img_cls = F.nll_loss(tgt_base_prob, tgt_base_label)


        fake_tgt_base_score, fake_tgt_base_label = self.target_D(fake_tgt_base_feat, need_backprop)
        fake_tgt_base_prob = F.log_softmax(fake_tgt_base_score, dim=1)
        fake_tgt_img_cls = F.nll_loss(fake_tgt_base_prob, fake_tgt_base_label)

        instance_loss = nn.BCELoss()
     
        source_ins_cls = 0
        fake_source_ins_cls = 0 
        target_ins_cls = 0
        fake_tgt_ins_cls = 0

        instance_sigmoid, same_size_label = self.source_instance(
            pooled_feat, need_backprop
        )
        source_ins_cls = instance_loss(instance_sigmoid, same_size_label)

        fake_instance_sigmoid, fake_same_size_label = self.source_instance(
            fake_pooled_feat, tgt_need_backprop
        )
        fake_source_ins_cls = instance_loss(fake_instance_sigmoid, fake_same_size_label)

        tgt_instance_sigmoid, tgt_same_size_label = self.target_instance(
            tgt_pooled_feat, need_backprop
        )
        target_ins_cls = instance_loss(tgt_instance_sigmoid, tgt_same_size_label)

        fake_tgt_instance_sigmoid, fake_tgt_same_size_label = self.target_instance(
            fake_tgt_pooled_feat, tgt_need_backprop
        )
        fake_tgt_ins_cls = instance_loss(fake_tgt_instance_sigmoid, fake_tgt_same_size_label)



        return (
            rois,
            cls_prob,
            bbox_pred,
            tgt_rois, 
            tgt_cls_prob,
            tgt_bbox_pred,
            rpn_loss_cls,
            rpn_loss_bbox,
            fake_tgt_rpn_loss_cls, 
            fake_tgt_rpn_loss_bbox, 
            RCNN_loss_cls,
            RCNN_loss_bbox,
            fake_tgt_RCNN_loss_cls, 
            fake_tgt_RCNN_loss_bbox, 
            #rois_label,
            #base_feat, 
            #tgt_base_feat,
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
            target_norm_loss, 
            source_ins_cls, 
            fake_source_ins_cls, 
            target_ins_cls, 
            fake_tgt_ins_cls
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
        normal_init(self.tgt_RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.tgt_RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.tgt_RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.tgt_RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.tgt_RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        #normal_init(self.source_D.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.source_D.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.target_D.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.target_D.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.source_instance.dc_ip1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.source_instance.dc_ip2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.source_instance.clssifer, 0, 0.05, cfg.TRAIN.TRUNCATED)
        #normal_init(self.target_instance.dc_ip1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.target_instance.dc_ip2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.target_instance.clssifer, 0, 0.05, cfg.TRAIN.TRUNCATED)
        #normal_init(self.adain.DA.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.adain.DA.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.adain_chaos.DA.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.adain_chaos.DA.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
