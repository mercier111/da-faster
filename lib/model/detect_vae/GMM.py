import torch 
import time
from torch.distributions import  MultivariateNormal, Normal
import random
import torch.nn as nn
import numpy as np 
import cv2
#from model.detect_vae.resnet import BasicBlock

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=True
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True  # change
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = out + self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return  out

class _to_detect_map(nn.Module):
    def __init__(self, class_num):
        super(_to_detect_map, self).__init__()
        self.class_num = class_num
        self.bottleneck = Bottleneck(3, class_num)
        self.softmax = nn.Softmax(2)
        self.eps = 1e-11

    def forward(self, x):
        x = self.bottleneck(x)
        N, C, H, W = x.shape

        x = x.view(N, C, H * W)
        map = self.softmax(x+self.eps).view(N, C, H, W).cuda()
        return map 

#size = torch.tensor([1, 3, 600, 800]).shape
#x = torch.ones(1, 3, 3, 4)
#map = _to_detect_map(100)(x)
#print(map.shape)

def norm_to_bbox():
    pass


def bbox_to_norm(bbox, img_size):
    x, y, w, h = bbox

    N, C, H, W = img_size

    mean = torch.tensor([x, y])
    var  = torch.tensor([[w, 0.], 
                         [0., h]])

    dist = MultivariateNormal(mean, var)

    detect_map = np.fromfunction(lambda i, j: dist.log_prob(torch.stack([torch.FloatTensor(i), torch.FloatTensor(j)],2)).exp(), (H, W))
    detect_map = nn.Softmax(2)(detect_map.view(N, 1, H, W)).view(H, W).cuda()
    return detect_map

class Random_detect_map(nn.Module):
    def __init__(self, class_num=1):
        super(Random_detect_map, self).__init__()
        self.class_num = class_num
        self.object_num = 3
        self.softmax = nn.Softmax(2)

    
    def random_box(self, img_size):
        N, C, H, W = img_size
        x_rand, y_rand = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
        x = int(x_rand * W)
        y = int(y_rand * H) 
        w = int(random.uniform(0.4, 0.8) * min(x, W-x))
        h = int(random.uniform(0.4, 0.8) * min(y, H-y))
        return x, y, w, h

    def forward(self, img_size, box_num=30):
        N, C, H, W = img_size 
        sum = torch.FloatTensor(torch.zeros(N, self.class_num, H, W)).cuda()
        gt_boxes = torch.FloatTensor(torch.zeros([1, 30, 5])).cuda()
        num_boxes = 0
        for i in range(self.class_num):
            object_num = random.choice(range(1, self.object_num+1))
            for j in range(object_num):
                bbox = self.random_box(img_size)
                x, y, w, h = bbox
                detect_map = bbox_to_norm(bbox, img_size)
                sum[0][i] += detect_map
                # 左上右下格式
                gt_boxes[0][num_boxes] = torch.FloatTensor([x-0.5*w, y-0.5*h, x+0.5*w, y+0.5*h, i+1])
                num_boxes += 1
                
            sum = nn.Softmax(2)(sum.view(N, self.class_num, H*W)).view(N, self.class_num, H, W)
            
        return sum, gt_boxes.cuda(), torch.LongTensor([num_boxes]).cuda()

class four2three(nn.Module):
    def __init__(self, classes):
        super(four2three, self).__init__()
        self.Conv = nn.Conv2d(3 + classes, 3, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.Conv(x)
        x + self.relu(x)
        return x 

class Encoder(nn.Module):
    def __init__(self, class_num, input_dim=3, other_dim=3):
        super(Encoder, self).__init__()
        self.class_num = int(class_num)
        self.other_dim = other_dim
        self.bottleneck = Bottleneck(input_dim, other_dim+class_num)
        self.bottleneck2 = Bottleneck(other_dim+class_num, other_dim + class_num)
        self.bottleneck3 = Bottleneck(other_dim+class_num, other_dim + class_num)
        self.softmax = nn.Softmax(2)

    def forward(self, x, domain=0):
        x = self.bottleneck(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        
        detect_map = x[:, :self.class_num, :, :]
        other_feat = x[:, self.class_num:, :, :]
        
        N, C, H, W = detect_map.shape
        detect_map = detect_map.view(N, C, H * W)
        detect_map = self.softmax(detect_map).view(N, C, H, W).cuda()
        #print(detect_map.shape, detect_map.mean(), "real_detect_map")
        if domain == 0:
            domain_map = torch.zeros(N, 1, H, W).cuda()
            other_feat = torch.cat((other_feat, domain_map), dim=1) 
        else: 
            domain_map = torch.ones(N, 1, H, W).cuda()
            other_feat = torch.cat((other_feat, domain_map), dim=1) 
        return detect_map, other_feat


class Decoder(nn.Module):
    def __init__(self, class_num, input_dim=3, other_dim=3):
        super(Decoder, self).__init__()
        self.class_num = int(class_num)
        self.bottleneck = Bottleneck(class_num * (other_dim + 1), input_dim)
        self.bottleneck2 = Bottleneck(input_dim, input_dim)
        self.bottleneck3 = Bottleneck(input_dim, input_dim)
        

    def forward(self, detect_map, other_feat):

        img = detect_map[0].cpu().detach().numpy() 
        img = (img - img.min()) * 255 /(img.max() - img.min()) 

        img = np.transpose(img.astype('uint8'), (1, 2, 0)).copy()
        cv2.imwrite('test.png', img)

        N, C1, H, W = detect_map.shape
        N, C2, H, W = other_feat.shape

        x = torch.zeros(N, C1 * C2, H, W).cuda()
        for i in range(C1):
            for j in range(C2):
                x[0][5*i + j] = detect_map[0][i] * other_feat[0][j]
        #x = torch.cat((detect_map, other_feat), dim=1)
        
        x = self.bottleneck(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
         

        return x

#detect_map = bbox_to_norm([100, 100, 100, 100], (1, 3, 500, 500))
#detect_map = detect_map.view(1, 1, 500, 500) * 255
 
#img = detect_map[0].cpu().detach().numpy() * 500 *500 *50
#
#img = np.transpose(img.astype('uint8'), (1, 2, 0)).copy()
#
#cv2.imwrite('test.png', img)

#mean = torch.tensor([100, 100])
#var  = torch.tensor([[20, 0.], 
#                     [0., 50]])
#
#dist = MultivariateNormal(mean, var)
#detect_map = np.fromfunction(lambda i, j: dist.log_prob(torch.stack([torch.FloatTensor(i), torch.FloatTensor(j)],2)).exp(), (200, 200)).view(1, 1, 200, 200)
#img = detect_map[0].cpu().detach().numpy() 
#img /= (img.max() - img.min())
#img *= 255

#sum = torch.FloatTensor(torch.zeros(1, 1, 500, 500)).cuda()
#map1 = bbox_to_norm([200, 100, 100, 100], [1, 1, 500, 500]).view(1, 1, 500, 500)
#map2 = bbox_to_norm([100, 300, 100, 100], [1, 1, 500, 500]).view(1, 1, 500, 500)
#
#img = map1+map2
#img = nn.Softmax(0)(img.view(500*500)).view(1, 1, 500, 500)
#img = img[0].cpu().detach().numpy() 
#img = map1[0].cpu().detach().numpy() 
#img = img + map2[0].cpu().detach().numpy() 
#print(img.min(), img.max())
#img = (img-img.min())/(img.max() - img.min())
#img = img * 255
#img = np.transpose(img.astype('uint8'), (1, 2, 0)).copy()
#
#cv2.imwrite('test.png', img)

#a = torch.rand(1, 3, 200, 200)
#b = torch.rand(1, 5, 200, 200)
#c = torch.zeros(1, 15, 200, 200)
#for i in range(3):
#    for j in range(5):
#       c[0][5*i + j] = a[0][i] * b[0][j]
#print(c.shape)