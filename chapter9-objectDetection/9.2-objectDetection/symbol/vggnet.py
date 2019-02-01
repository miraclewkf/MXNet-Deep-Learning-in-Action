import mxnet as mx

def VGGNet():
    data = mx.sym.Variable(name='data')

    conv1_1 = mx.sym.Convolution(data=data, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=64, name='conv1_1')
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type='relu', name='relu1_1')
    conv1_2 = mx.sym.Convolution(data=relu1_1, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=64, name='conv1_2')
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type='relu', name='relu1_2')
    pool1 = mx.sym.Pooling(data=relu1_2, kernel=(2,2), stride=(2,2), pool_type='max',
                           pooling_convention='full', name='pool1')

    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=128, name='conv2_1')
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type='relu', name='relu2_1')
    conv2_2 = mx.sym.Convolution(data=relu2_1, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=128, name='conv2_2')
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type='relu', name='relu2_2')
    pool2 = mx.sym.Pooling(data=relu2_2, kernel=(2,2), stride=(2,2), pool_type='max',
                           pooling_convention='full', name='pool2')

    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=256, name='conv3_1')
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type='relu', name='relu3_1')
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=256, name='conv3_2')
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type='relu', name='relu3_2')
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                 num_filter=256, name='conv3_3')
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type='relu', name='relu3_3')
    pool3 = mx.sym.Pooling(data=relu3_3, kernel=(2, 2), stride=(2, 2), pool_type='max',
                           pooling_convention='full', name='pool3')

    conv4_1 = mx.sym.Convolution(data=pool3, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=512, name='conv4_1')
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type='relu', name='relu4_1')
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=512, name='conv4_2')
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type='relu', name='relu4_2')
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=512, name='conv4_3')
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type='relu', name='relu4_3')
    pool4 = mx.sym.Pooling(data=relu4_3, kernel=(2,2), stride=(2,2), pool_type='max',
                           pooling_convention='full', name='pool4')

    conv5_1 = mx.sym.Convolution(data=pool4, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=512, name='conv5_1')
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type='relu', name='relu5_1')
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=512, name='conv5_2')
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type='relu', name='relu5_2')
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3,3), pad=(1,1), stride=(1,1),
                                 num_filter=512, name='conv5_3')
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type='relu', name='relu5_3')
    pool5 = mx.sym.Pooling(data=relu5_3, kernel=(3,3), pad=(1,1), stride=(1,1),
                           pool_type='max', pooling_convention='full', name='pool5')

    conv6 = mx.sym.Convolution(data=pool5, kernel=(3,3), pad=(6,6), stride=(1,1),
                               num_filter=1024, dilate=(6,6), name='fc6')
    relu6 = mx.sym.Activation(data=conv6, act_type='relu', name='relu6')

    conv7 = mx.sym.Convolution(data=relu6, kernel=(1,1), stride=(1,1),
                               num_filter=1024, name='fc7')
    relu7 = mx.sym.Activation(data=conv7, act_type='relu', name='relu7')
    return relu7