from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt
import mxnet as mx

ssd = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=True)
img_path = "demo_img/2007_001311.jpg"
img = mx.image.imread(filename=img_path).asnumpy().astype('uint8')
print("Shape of original image: {}".format(img.shape))
data, img = data.transforms.presets.ssd.load_test(filenames=img_path,
                                                  short=300)
print("Shape of resize image: {}".format(img.shape))
print("Shape of transform image: {}".format(data.shape))

labels,scores,bboxes = ssd(data)
print("Shape of predict labels: {}".format(labels.shape))
print("Shape of predict scores: {}".format(scores.shape))
print("Shape of predict bboxes: {}".format(bboxes.shape))

utils.viz.plot_bbox(img=img,
                    bboxes=bboxes[0],
                    scores=scores[0],
                    labels=labels[0],
                    class_names=ssd.classes)
plt.savefig('object_detection_result.png')
