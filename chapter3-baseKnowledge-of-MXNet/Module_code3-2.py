import mxnet as mx
import logging

data = mx.sym.Variable('data')
conv = mx.sym.Convolution(data=data, num_filter=128, kernel=(3,3), pad=(1,1), 
                          name='conv1')
bn = mx.sym.BatchNorm(data=conv, name='bn1')
relu = mx.sym.Activation(data=bn, act_type='relu', name='relu1')
pool = mx.sym.Pooling(data=relu, kernel=(2,2), stride=(2,2), pool_type='max', 
                      name='pool1')
fc = mx.sym.FullyConnected(data=pool, num_hidden=2, name='fc1')
sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')

data = mx.nd.random.uniform(0,1,shape=(1000,3,224,224))
label = mx.nd.round(mx.nd.random.uniform(0,1,shape=(1000)))
train_data = mx.io.NDArrayIter(data={'data':data},
                               label={'softmax_label':label},
                               batch_size=8,
                               shuffle=True)

print(train_data.provide_data)
print(train_data.provide_label)
mod = mx.mod.Module(symbol=sym,context=mx.gpu(0))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
mod.fit(train_data=train_data, num_epoch=5)

