from __future__ import absolute_import, division, print_function
from turtle import forward

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.utils.config import cfg
from torch.autograd import Function, Variable
from model.cycle_faster_rcnn.DA import _ImageDA
from model.cycle_faster_rcnn.GMM import bbox_to_norm, Random_detect_map

class Adain(nn.Module):
    def __init__(self, eps=1e-5):
        super(Adain, self).__init__()
        self.eps = eps

    def forward(self, source_feat, target_feat):

        N, C, H, W = source_feat.size()
        _N, _C, _H, _W = target_feat.size()

        source_feat = source_feat.view(N, C, -1)
        source_mean = source_feat.mean(-1, keepdim=True)
        source_var  = source_feat.var(-1, keepdim=True)
        source_norm = (source_feat - source_mean) / (source_var + self.eps).sqrt()

        target_feat = target_feat.view(_N, _C, -1)
        target_mean = target_feat.mean(-1, keepdim=True)
        target_var  = target_feat.var(-1, keepdim=True)
        target_norm = (target_feat - target_mean) / (target_var + self.eps).sqrt()
        
        fake_source_feat = target_norm * (source_var + self.eps).sqrt() + source_mean
        fake_source_feat = fake_source_feat.view(_N, _C, _H, _W)

        fake_target_feat = source_norm * (target_var + self.eps).sqrt() + target_mean
        fake_target_feat = fake_target_feat.view(N, C, H, W)

        source_skews = torch.abs(torch.mean(torch.pow(source_norm, 3.0)))
        source_kurtoses = torch.abs(torch.mean(torch.pow(source_norm, 4.0)) - 3.0)

        target_skews = torch.abs(torch.mean(torch.pow(target_norm, 3.0)))
        target_kurtoses = torch.abs(torch.mean(torch.pow(target_norm, 4.0)) - 3.0)

        return fake_source_feat, fake_target_feat, source_skews, target_skews, source_kurtoses, target_kurtoses



class Adain_with_da(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(Adain_with_da, self).__init__()
        self.eps = eps
        self.DA = _ImageDA(dim)
        self.need_backprop = torch.Tensor([1]).cuda()
        self.tgt_need_backprop = torch.Tensor([0]).cuda()

    def forward(self, source_feat, target_feat):
        
        N, C, H, W = source_feat.size()
        _N, _C, _H, _W = target_feat.size()

        source_feat = source_feat.view(N, C, -1)
        source_mean = source_feat.mean(-1, keepdim=True)
        source_var  = source_feat.var(-1, keepdim=True)
        source_norm = (source_feat - source_mean) / (source_var + self.eps).sqrt()
        
        target_feat = target_feat.view(_N, _C, -1)
        target_mean = target_feat.mean(-1, keepdim=True)
        target_var  = target_feat.var(-1, keepdim=True)
        target_norm = (target_feat - target_mean) / (target_var + self.eps).sqrt()
        
        fake_source_feat = target_norm * (source_var + self.eps).sqrt() + source_mean
        fake_source_feat = fake_source_feat.view(_N, _C, _H, _W)

        fake_target_feat = source_norm * (target_var + self.eps).sqrt() + target_mean
        fake_target_feat = fake_target_feat.view(N, C, H, W)
    
        source_skews = torch.abs(torch.mean(torch.pow(source_norm, 3.0)))
        source_kurtoses = torch.abs(torch.mean(torch.pow(source_norm, 4.0)) - 3.0)

        target_skews = torch.abs(torch.mean(torch.pow(target_norm, 3.0)))
        target_kurtoses = torch.abs(torch.mean(torch.pow(target_norm, 4.0)) - 3.0)
        
        base_score, base_label = self.DA(source_norm.view(N, C, H, W), self.need_backprop)
        base_prob = F.log_softmax(base_score, dim=1)
        source_norm_loss = F.nll_loss(base_prob, base_label)

        tgt_base_score, tgt_base_label = self.DA(target_norm.view(_N, _C, _H, _W), self.tgt_need_backprop)
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        target_norm_loss = F.nll_loss(tgt_base_prob, tgt_base_label)

        return fake_source_feat, fake_target_feat, source_skews, target_skews, source_kurtoses, target_kurtoses, source_norm_loss, target_norm_loss

class Adain_with_chaos(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(Adain_with_chaos, self).__init__()
        self.eps = eps
        self.DA = _ImageDA(dim)
        self.need_backprop = torch.Tensor([1]).cuda()
        self.tgt_need_backprop = torch.Tensor([0]).cuda()

        def get_norm(feat, mean, var, eps=self.eps):
            #norm = (feat - var) / (mean +eps).sqrt()
            #norm = (feat + var ) * (mean +eps).sqrt()
            norm = feat * mean
            #norm = feat * var
            return norm 

        def from_norm(norm, mean, var, eps=self.eps):
            #feat = norm * (mean + self.eps).sqrt() + var
            #feat = norm / (feat + var ) / (mean +eps).sqrt()
            feat = norm / mean
            #feat = norm / var 
            return feat

        self.get_norm = get_norm
        self.from_norm = from_norm

    def forward(self, source_feat, target_feat):
        
        N, C, H, W = source_feat.size()
        _N, _C, _H, _W = target_feat.size()

        source_feat = source_feat.view(N, C, -1)
        source_mean = source_feat.mean(-1, keepdim=True)
        source_var  = source_feat.var(-1, keepdim=True)
        #source_norm = (source_feat - source_var) / (source_mean + self.eps).sqrt()
        source_norm = self.get_norm(source_feat, source_mean, source_var)

        target_feat = target_feat.view(_N, _C, -1)
        target_mean = target_feat.mean(-1, keepdim=True)
        target_var  = target_feat.var(-1, keepdim=True)
        #target_norm = (target_feat - target_var) / (target_mean + self.eps).sqrt()
        target_norm = self.get_norm(target_feat, target_mean, target_var)


        fake_source_feat = target_norm * (source_mean + self.eps).sqrt() + source_var
        #fake_source_feat = self.from_norm(target_norm, source_mean, source_var)
        fake_source_feat = fake_source_feat.view(_N, _C, _H, _W)

        fake_target_feat = source_norm * (target_mean + self.eps).sqrt() + target_var
        #fake_target_feat = self.from_norm(source_norm, target_mean, target_var)
        fake_target_feat = fake_target_feat.view(N, C, H, W)
    
        source_skews = torch.abs(torch.mean(torch.pow(source_norm, 3.0)))
        source_kurtoses = torch.abs(torch.mean(torch.pow(source_norm, 4.0)) - 3.0)

        target_skews = torch.abs(torch.mean(torch.pow(target_norm, 3.0)))
        target_kurtoses = torch.abs(torch.mean(torch.pow(target_norm, 4.0)) - 3.0)
        
        base_score, base_label = self.DA(source_norm.view(N, C, H, W), self.need_backprop)
        base_prob = F.log_softmax(base_score, dim=1)
        source_norm_loss = F.nll_loss(base_prob, base_label)

        tgt_base_score, tgt_base_label = self.DA(target_norm.view(_N, _C, _H, _W), self.tgt_need_backprop)
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        target_norm_loss = F.nll_loss(tgt_base_prob, tgt_base_label)

        return fake_source_feat, fake_target_feat, source_skews, target_skews, source_kurtoses, target_kurtoses, source_norm_loss, target_norm_loss



class Adain_with_chaos(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(Adain_with_chaos, self).__init__()
        self.eps = eps
        self.DA = _ImageDA(dim)
        self.need_backprop = torch.Tensor([1]).cuda()
        self.tgt_need_backprop = torch.Tensor([0]).cuda()

        def get_norm(feat, mean, var, eps=self.eps):
            norm = (feat - mean) / (var +eps).sqrt()
            #norm = (feat + var ) * (mean +eps).sqrt()
            #norm = feat * mean
            #norm = feat * var
            return norm 

        def from_norm(norm, mean, var, eps=self.eps):
            feat = norm * (var + self.eps).sqrt() + mean
            #feat = norm / (feat + var ) / (mean +eps).sqrt()
            #feat = norm / mean
            #feat = norm / var 
            return feat

        self.get_norm = get_norm
        self.from_norm = from_norm

    def forward(self, source_feat, target_feat):
        
        N, C, H, W = source_feat.size()
        _N, _C, _H, _W = target_feat.size()

        source_feat = source_feat.view(N, C, -1)
        source_mean = source_feat.mean(-1, keepdim=True)
        source_var  = source_feat.var(-1, keepdim=True)

        source_norm = self.get_norm(source_feat, source_mean, source_var)

        target_feat = target_feat.view(_N, _C, -1)
        target_mean = target_feat.mean(-1, keepdim=True)
        target_var  = target_feat.var(-1, keepdim=True)

        target_norm = self.get_norm(target_feat, target_mean, target_var)


        fake_source_feat = target_norm * (source_mean + self.eps).sqrt() + source_var
        fake_source_feat = fake_source_feat.view(_N, _C, _H, _W)

        fake_target_feat = source_norm * (target_mean + self.eps).sqrt() + target_var
        fake_target_feat = fake_target_feat.view(N, C, H, W)
    
        source_skews = torch.abs(torch.mean(torch.pow(source_norm, 3.0)))
        source_kurtoses = torch.abs(torch.mean(torch.pow(source_norm, 4.0)) - 3.0)

        target_skews = torch.abs(torch.mean(torch.pow(target_norm, 3.0)))
        target_kurtoses = torch.abs(torch.mean(torch.pow(target_norm, 4.0)) - 3.0)
        
        base_score, base_label = self.DA(source_norm.view(N, C, H, W), self.need_backprop)
        base_prob = F.log_softmax(base_score, dim=1)
        source_norm_loss = F.nll_loss(base_prob, base_label)

        tgt_base_score, tgt_base_label = self.DA(target_norm.view(_N, _C, _H, _W), self.tgt_need_backprop)
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        target_norm_loss = F.nll_loss(tgt_base_prob, tgt_base_label)

        return fake_source_feat, fake_target_feat, source_skews, target_skews, source_kurtoses, target_kurtoses, source_norm_loss, target_norm_loss


class get_stat(nn.Module):
    def __init__(self, dim=1024, eps=1e-5):
        super(get_stat, self).__init__()
        self.dim = dim 

        self.mean_linear = nn.Linear(dim, 256)
        self.var_linear = nn.Linear(dim, 256)
        self.mean_linear2 = nn.Linear(256, dim)
        self.var_linear2 = nn.Linear(256, dim)

        self.mean_relu1 = nn.ReLU()
        self.var_relu1 = nn.ReLU()
        self.mean_relu2 = nn.ReLU()
        self.var_relu2 = nn.ReLU()
    
    def forward(self, source_feat, target_feat):
        torch.autograd.set_detect_anomaly(True)
        mean = source_feat.mean(-1, keepdim=True)
        _mean = self.mean_relu1(self.mean_linear(mean.view(-1, self.dim)))
        _mean = torch.sigmoid(self.mean_relu2(self.mean_linear2(_mean)))
        _mean = _mean.unsqueeze_(-1)

        var = source_feat.var(-1, keepdim=True)
        _var = self.var_relu1(self.var_linear(var.view(-1, self.dim)))
        _var = torch.sigmoid(self.var_relu2(self.var_linear2(_var)))
        _var = _var.unsqueeze_(-1)

        tgt_mean = target_feat.mean(-1, keepdim=True)
        tgt_mean = self.mean_relu1(self.mean_linear(tgt_mean.view(-1, self.dim)))
        tgt_mean = torch.sigmoid(self.mean_relu2(self.mean_linear2(tgt_mean)))
        tgt_mean = _mean.unsqueeze_(-1)

        tgt_var = target_feat.var(-1, keepdim=True)
        tgt_var = self.var_relu1(self.var_linear(tgt_var.view(-1, self.dim)))
        tgt_var = torch.sigmoid(self.var_relu2(self.var_linear2(tgt_var)))
        tgt_var = _var.unsqueeze_(-1)

        return _mean, _var, tgt_mean, tgt_var