import mxnet as mx

def vgg16_pool3(input):
    conv1_1 = mx.sym.Convolution(data=input, kernel=(3,3), pad=(100,100),
                                 num_filter=64, name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu",
                                name="relu1_1")
    conv1_2 = mx.sym.Convolution(data=relu1_1, kernel=(3,3), pad=(1,1),
                                 num_filter=64, name="conv1_2")
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu",
                                name="relu1_2")
    pool1 = mx.sym.Pooling(data=relu1_2, pool_type="max", kernel=(2,2),
                           stride=(2,2), name="pool1")

    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3,3), pad=(1,1),
                                 num_filter=128, name="conv2_1")
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu",
                                name="relu2_1")
    conv2_2 = mx.sym.Convolution(data=relu2_1, kernel=(3,3), pad=(1,1),
                                 num_filter=128, name="conv2_2")
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu",
                                name="relu2_2")
    pool2 = mx.sym.Pooling(data=relu2_2, pool_type="max", kernel=(2,2),
                           stride=(2,2), name="pool2")

    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3,3), pad=(1,1),
                                 num_filter=256, name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu",
                                name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(3,3), pad=(1,1),
                                 num_filter=256, name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu",
                                name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3,3), pad=(1,1),
                                 num_filter=256, name="conv3_3")
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu",
                                name="relu3_3")
    pool3 = mx.sym.Pooling(data=relu3_3, pool_type="max", kernel=(2,2),
                           stride=(2,2), name="pool3")
    return pool3

def vgg16_pool4(input):
    conv4_1 = mx.sym.Convolution(data=input, kernel=(3,3), pad=(1,1),
                                 num_filter=512, name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu",
                                name="relu4_1")
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(3,3), pad=(1,1),
                                 num_filter=512, name="conv4_2")
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu",
                                name="relu4_2")
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3,3), pad=(1,1),
                                 num_filter=512, name="conv4_3")
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu",
                                name="relu4_3")
    pool4 = mx.sym.Pooling(data=relu4_3, pool_type="max", kernel=(2,2),
                           stride=(2,2), name="pool4")
    return pool4

def vgg16_score(input, num_classes):
    conv5_1 = mx.sym.Convolution(data=input, kernel=(3,3), pad=(1,1),
                                 num_filter=512, name="conv5_1")
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu",
                                name="relu5_1")
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(3,3), pad=(1,1),
                                 num_filter=512, name="conv5_2")
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu",
                                name="relu5_2")
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3,3), pad=(1,1),
                                 num_filter=512, name="conv5_3")
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu",
                                name="relu5_3")
    pool5 = mx.sym.Pooling(data=relu5_3, pool_type="max", kernel=(2,2),
                           stride=(2,2), name="pool5")

    fc6 = mx.sym.Convolution(data=pool5, kernel=(7,7), num_filter=4096,
                             name="fc6")
    relu6 = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")

    fc7 = mx.sym.Convolution(data=drop6, kernel=(1,1), num_filter=4096,
                             name="fc7")
    relu7 = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")

    weight_score = mx.sym.Variable(name="score_weight",
                                   init=mx.init.Constant(0))
    score = mx.sym.Convolution(data=drop7, kernel=(1,1), weight=weight_score,
                               num_filter=num_classes, name="score")
    return score

def fcnxs_score(input, crop, offset, kernel, stride, num_classes):
    bigscore = mx.sym.Deconvolution(data=input, kernel=kernel, stride=stride,
                                    adj=(stride[0]-1, stride[1]-1),
                                    num_filter=num_classes, name="bigscore")
    upscore = mx.sym.Crop(*[bigscore, crop], offset=offset, name="upscore")
    softmax = mx.sym.SoftmaxOutput(data=upscore, multi_output=True,
                                   use_ignore=True, ignore_label=-1,
                                   name="softmax", normalization="valid")
    return softmax

def symbol_fcn32s(num_classes=21):
    data = mx.sym.Variable(name="data")
    pool3 = vgg16_pool3(data)
    pool4 = vgg16_pool4(pool3)
    score = vgg16_score(pool4, num_classes)
    softmax = fcnxs_score(score, data, offset=(19,19), kernel=(64,64),
                          stride=(32,32), num_classes=num_classes)
    return softmax

def symbol_fcn16s(num_classes=21):
    data = mx.sym.Variable(name="data")
    pool3 = vgg16_pool3(data)
    pool4 = vgg16_pool4(pool3)
    score = vgg16_score(pool4, num_classes)

    score2 = mx.sym.Deconvolution(data=score, kernel=(4,4),
                                  stride=(2,2), num_filter=num_classes,
                                  adj=(1,1), name="score2")
    weight_score_pool4 = mx.sym.Variable(name="score_pool4_weight",
                                         init=mx.init.Constant(0))
    score_pool4 = mx.sym.Convolution(data=pool4, kernel=(1,1),
                                     weight=weight_score_pool4,
                                     num_filter=num_classes,
                                     name="score_pool4")
    score_pool4c = mx.sym.Crop(*[score_pool4, score2], offset=(5,5),
                               name="score_pool4c")
    score_fused = score2 + score_pool4c
    softmax = fcnxs_score(score_fused, data, offset=(27,27), kernel=(32,32),
                          stride=(16,16), num_classes=num_classes)
    return softmax

def symbol_fcn8s(num_classes=21):
    data = mx.sym.Variable(name="data")
    pool3 = vgg16_pool3(data)
    pool4 = vgg16_pool4(pool3)
    score = vgg16_score(pool4, num_classes)

    score2 = mx.sym.Deconvolution(data=score, kernel=(4,4),
                                  stride=(2,2), num_filter=num_classes,
                                  adj=(1,1), name="score2")
    weight_score_pool4 = mx.sym.Variable(name="score_pool4_weight",
                                         init=mx.init.Constant(0))
    score_pool4 = mx.sym.Convolution(data=pool4, kernel=(1,1),
                                     weight=weight_score_pool4,
                                     num_filter=num_classes,
                                     name="score_pool4")
    score_pool4c = mx.sym.Crop(*[score_pool4, score2], offset=(5,5),
                               name="score_pool4c")
    score_fused = score2 + score_pool4c

    score4 = mx.sym.Deconvolution(data=score_fused, kernel=(4,4),
                                  stride=(2,2), num_filter=num_classes,
                                  adj=(1,1), name="score4")
    weight_score_pool3 = mx.sym.Variable(name="score_pool3_weight",
                                         init=mx.init.Constant(0))
    score_pool3 = mx.sym.Convolution(data=pool3, kernel=(1,1),
                                     weight=weight_score_pool3,
                                     num_filter=num_classes,
                                     name="score_pool3")
    score_pool3c = mx.sym.Crop(*[score_pool3, score4], offset=(9,9),
                               name="score_pool3c")
    score_final = score4 + score_pool3c
    softmax = fcnxs_score(score_final, data, offset=(31,31), kernel=(16,16),
                          stride=(8,8), num_classes=num_classes)
    return softmax