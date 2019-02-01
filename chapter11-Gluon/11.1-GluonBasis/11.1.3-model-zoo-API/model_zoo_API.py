import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision

resnet18_v1 = vision.resnet18_v1(pretrained=False)
print(resnet18_v1.output.weight._data)

resnet18_v1 = vision.resnet18_v1(pretrained=True)
print(resnet18_v1.output.weight._data)
data = mx.nd.random.uniform(0,1,(1,3,224,224))
output = resnet18_v1.forward(data)
print("Shape of output is: {}".format(output.shape))

resnet18_v1.output = nn.Dense(5)
resnet18_v1.output.initialize()
output = resnet18_v1(data)
print("Shape of output is: {}".format(output.shape))