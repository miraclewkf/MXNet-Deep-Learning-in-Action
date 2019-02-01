import mxnet as mx
from mxnet.gluon import nn
import time

class Bottleneck(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.body = nn.HybridSequential()
        self.body.add(nn.Conv2D(channels=64, kernel_size=1),
                      nn.BatchNorm(),
                      nn.Activation(activation='relu'),
                      nn.Conv2D(channels=64, kernel_size=3, padding=1),
                      nn.BatchNorm(),
                      nn.Activation(activation='relu'),
                      nn.Conv2D(channels=256, kernel_size=1),
                      nn.BatchNorm())

    def hybrid_forward(self,F, x):
        residual = x
        x = self.body(x)
        x = F.Activation(x+residual, act_type='relu')
        return x

data = mx.nd.random.uniform(1,5,shape=(2,256,224,224), ctx=mx.gpu(0))
net1 = Bottleneck()
net1.initialize(ctx=mx.gpu(0))
t1 = time.time()
output = net1(data)
t2 = time.time()
print("Dynamic graph forward time: {:.4f}ms".format((t2-t1)*1000))

net2 = Bottleneck()
net2.initialize(ctx=mx.gpu(0))
net2.hybridize()
t3 = time.time()
output = net2(data)
t4 = time.time()
print("Static graph forward time: {:.4f}ms".format((t4-t3)*1000))
