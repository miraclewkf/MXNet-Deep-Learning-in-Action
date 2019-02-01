from matplotlib import pyplot as plt
from gluoncv import data, utils

train_data = data.VOCDetection(root='data/VOCdevkit',
                               splits=[(2007, 'trainval'),(2012,'trainval')])
val_data = data.VOCDetection(root='data/VOCdevkit',
                             splits=[(2007, 'test')])

# class names of Pascal VOC dataset
class_names = train_data.classes
print("Class names of Pascal VOC Detection: {}".format(class_names))

# Number of data
print("Number of train data: {}".format(len(train_data)))
print("Number of validation data: {}".format(len(val_data)))

# One image
image, ground_truth = train_data[0]
bbox = ground_truth[:,0:4]
label = ground_truth[:,4:5]
print("Shape of image: {}".format(image.shape))
print("Bounding box: {}".format(bbox))
print("Label of bbox: {}".format(label))
print("Label name of bbox: {}".format([class_names[int(i)] for i in label]))

# Visualization
utils.viz.plot_bbox(img=image,
                    bboxes=bbox,
                    labels=label,
                    class_names=class_names)
plt.savefig('object_detection_gt.png')
