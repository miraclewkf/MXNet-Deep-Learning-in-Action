import mxnet as mx
from gluoncv import model_zoo, data
from mxnet.gluon.data.vision import transforms
import matplotlib.pyplot as plt

resnet50 = model_zoo.get_model('resnet50_v2',pretrained=True)
img = mx.image.imread(filename="demo_img/ILSVRC2012_val_00003559.JPEG")
print("Shape of original image: {}".format(img.shape))

transform_size = transforms.Compose([transforms.Resize(256, keep_ratio=True),
                                     transforms.CenterCrop(224)])
img = transform_size(img)
plt.imshow(img.asnumpy())
plt.savefig('transform_result.png')

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
transform_fn = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=mean,
                                                        std=std)])
data = transform_fn(img).expand_dims(0)
print("Shape of transform image: {}".format(data.shape))

output = resnet50.forward(data)
top5_index = mx.nd.topk(output, k=5)[0].astype('int')
print("Predict label for input image is:")
for index_i in top5_index:
    pre_label = resnet50.classes[index_i.asscalar()]
    print(pre_label)