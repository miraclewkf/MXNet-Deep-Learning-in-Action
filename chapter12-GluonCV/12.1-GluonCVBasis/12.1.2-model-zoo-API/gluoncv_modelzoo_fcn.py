from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt
import mxnet as mx

fcn = model_zoo.get_model('fcn_resnet101_voc',pretrained=True)
img = mx.image.imread(filename="demo_img/2007_001311.jpg")
print("Shape of original image: {}".format(img.shape))
fig = plt.figure("segmentation")
fig.add_subplot(1,2,1)
plt.imshow(img.asnumpy().astype('uint8'))

data = mx.nd.image.to_tensor(img)
data = mx.nd.image.normalize(data,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
data = data.expand_dims(0)
print("Shape of transform image: {}".format(data.shape))

output = fcn.demo(data)
mask = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
mask = utils.viz.get_color_pallete(npimg=mask,
                                   dataset='pascal_voc')
fig.add_subplot(1,2,2)
plt.imshow(mask)
plt.savefig('segmentation_result.png')
