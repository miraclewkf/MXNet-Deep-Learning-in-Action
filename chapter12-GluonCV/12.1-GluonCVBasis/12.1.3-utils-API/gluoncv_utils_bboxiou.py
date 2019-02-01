import numpy as np
import gluoncv

bbox_a = np.array([[0,0,0.8,0.8]])
bbox_b = np.array([[0.4,0.4,1,1]])
iou = gluoncv.utils.bbox_iou(bbox_a, bbox_b)
print("IOU of bbox_a and bbox_b is: {}".format(iou[0][0]))
