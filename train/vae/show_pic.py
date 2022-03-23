import cv2
import numpy as np 

def tensor2numpy(img, gt_boxes=0, path=''):

    img = img[0].cpu().detach().numpy() 
    img = np.transpose(img.astype('uint8'), (1, 2, 0)).copy()
    
    if type(gt_boxes) != int:
        gt_boxes = gt_boxes[0].cpu().detach().numpy() 
        for bbox in gt_boxes:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])) , (0, 255, 0), 4)
    if path != '':
        cv2.imwrite(path, img)
    return img 