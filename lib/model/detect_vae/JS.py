import torch 
import torch.nn as nn
from model.detect_vae.GMM import bbox_to_norm
import torch.nn.functional as F

def kl_map_loss(detect_map, gt_boxes, num_boxes):
    N, C, H, W = detect_map.shape

    sum = torch.FloatTensor(torch.zeros_like(detect_map)).cuda()
    object_list = [0 for i in range(C)]
    for i in range(num_boxes):
        
        x_min, y_min, x_max, y_max, c = gt_boxes[i]
        object_list[c-1] += 1

        x = (x_min + x_max)/2
        y = (y_min + y_max)/2
        w = x_max - x_min 
        h = y_max - y_min 
        detect_map = bbox_to_norm((x, y, w, h) , detect_map.shape)
        sum[0][c-1] += detect_map
    
    for j in range(C):
        sum[0][c-1] /= object_list[c-1]
    
    #sum vs detectmap 
    kl_loss = detect_map * torch.log(detect_map/sum).sum()
    
    return kl_loss

def js_map_loss(detect_map, gt_boxes, num_boxes):

    N, C, H, W = detect_map.shape
    sum = torch.FloatTensor(torch.zeros_like(detect_map.to('cpu'))).cuda()
    object_list = [0 for i in range(C)]
    for i in range(num_boxes):
        x_min, y_min, x_max, y_max, c = gt_boxes[0][i]
        object_list[int(c)-1] += 1

        x = (x_min + x_max)/2
        y = (y_min + y_max)/2
        w = x_max - x_min 
        h = y_max - y_min 

        map = bbox_to_norm((x, y, w, h) , detect_map.shape)
        sum[0][int(c)-1] += map
    
    for j in range(C):
        sum[0][j] /= object_list[j]

    sum = sum.cuda()
    #sum vs detectmap 
    eps = torch.tensor([1e-5]).cuda()
    #js_loss = detect_map * torch.log(detect_map/(sum + eps)) + sum * torch.log(sum/(detect_map + eps))
    #js_loss = js_loss.mean()
    #js_loss = (detect_map - sum).abs().sum()
    
    js_loss = 0
    for i in range(C):
        js_loss += F.kl_div(torch.log(detect_map[0][i]), sum[0][i], reduction='mean')
        js_loss += F.kl_div(torch.log(sum[0][i]), detect_map[0][i], reduction='mean')

    return js_loss
#softmax = nn.Softmax(2)
#a = torch.ones(1, 1, 20, 20)
#b = torch.rand(1, 1, 20, 20)
#a = softmax(a)
#b = softmax(b)
#l = F.kl_div(torch.log(a), b)
#print(l)
#import torch.nn as nn
#a = torch.rand([3, 6, 8])
#b = torch.rand([3, 6, 8])
#soft = nn.Softmax(1)
#a = soft(a.view(3, 6*8)).view(3, 6, 8)
#b = soft(b.view(3, 6*8)).view(3, 6, 8)
#import torch.nn.functional as F
#js_loss = F.kl_div(torch.log(a), b, reduction='sum')
#print(a)
#a = F.softmax(a,dim=[1, 2])
#print(a)
#print(js_loss)

a = torch.ones(1, 3, 600, 800)
b = torch.ones(1, 3, 600, 800) * 256
l = nn.L1Loss(reduction='mean')(a, b)
print(l)