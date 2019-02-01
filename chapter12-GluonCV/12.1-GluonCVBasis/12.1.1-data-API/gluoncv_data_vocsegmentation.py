from matplotlib import pyplot as plt
from gluoncv import data, utils
import numpy as np

train_data = data.VOCSegmentation(root='data/VOCdevkit',
                                  split='train')
val_data = data.VOCSegmentation(root='data/VOCdevkit',
                                split='val')

# class names of Pascal VOC dataset
class_names = train_data.classes
print("Class names of Pascal VOC Segmentation: {}".format(class_names))

# Number of data
print("Number of train data: {}".format(len(train_data)))
print("Number of validation data: {}".format(len(val_data)))

# Visualization
image, ground_truth = train_data[0]
fig = plt.figure("segmentation")
fig.add_subplot(1,2,1)
plt.imshow(image.asnumpy().astype(np.uint8))
mask = utils.viz.get_color_pallete(npimg=ground_truth.asnumpy(),
                                   dataset='pascal_voc')
fig.add_subplot(1,2,2)
plt.imshow(mask)
plt.savefig('segmentation_gt.png')
