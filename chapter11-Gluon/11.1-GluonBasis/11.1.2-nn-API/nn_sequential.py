import mxnet as mx
from mxnet.gluon import nn

fc = nn.Dense(2)
fc.initialize()
data = mx.nd.random.uniform(1,5,(4,3))
print("Input data:")
print(data)
output = fc(data)
print("FC layer result:")
print(output)
print("FC layer weight:")
print(fc.weight.data())

############################# mxnet.gluon.nn.Sequential() #########################
import mxnet as mx
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.initialize()
data = mx.nd.random.uniform(1,5,shape=(2,1,28,28))
output = net(data)



