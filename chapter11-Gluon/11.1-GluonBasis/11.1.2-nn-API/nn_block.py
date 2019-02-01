import mxnet as mx
from mxnet.gluon import nn

class Bottleneck(nn.Block):
    def __init__(self, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.body = nn.Sequential()
        self.body.add(nn.Conv2D(channels=64, kernel_size=1),
                      nn.BatchNorm(),
                      nn.Activation(activation='relu'),
                      nn.Conv2D(channels=64, kernel_size=3, padding=1),
                      nn.BatchNorm(),
                      nn.Activation(activation='relu'),
                      nn.Conv2D(channels=256, kernel_size=1),
                      nn.BatchNorm())
        self.relu = nn.Activation(activation='relu')

    def forward(self, x):
        residual = x
        x = self.body(x)
        x = self.relu(x + residual)
        return x

net = Bottleneck()
net.initialize()
data = mx.nd.random.uniform(1,5,shape=(2,256,224,224))
output = net(data)