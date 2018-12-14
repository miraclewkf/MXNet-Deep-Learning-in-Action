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
mod.bind(data_shapes=train_data.provide_data,
         label_shapes=train_data.provide_label)
mod.init_params()
mod.init_optimizer()
eval_metric = mx.metric.create('acc')
for epoch in range(5):
    end_of_batch = False
    eval_metric.reset()
    data_iter = iter(train_data)
    next_data_batch = next(data_iter)
    while not end_of_batch:
        data_batch = next_data_batch
        mod.forward(data_batch)
        mod.backward()
        mod.update()
        mod.update_metric(eval_metric, labels=data_batch.label)
        try:
            next_data_batch = next(data_iter)
            mod.prepare(next_data_batch)
        except StopIteration:
            end_of_batch = True
    eval_name_vals = eval_metric.get_name_value()
    print("Epoch:{} Train_Acc:{:.4f}".format(epoch, eval_name_vals[0][1]))
    arg_params, aux_params = mod.get_params()
    mod.set_params(arg_params, aux_params)
    train_data.reset()

