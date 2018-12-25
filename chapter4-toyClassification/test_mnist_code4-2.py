import mxnet as mx
import numpy as np

def load_model(model_prefix, index, context, data_shapes, label_shapes):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, index)
    model = mx.mod.Module(symbol=sym, context=context)
    model.bind(data_shapes=data_shapes, label_shapes=label_shapes,
               for_training=False)
    model.set_params(arg_params=arg_params, aux_params=aux_params,
                     allow_missing=True)
    return model

def load_data(data_path):
    data = mx.image.imread(data_path, flag=0)
    cla_cast_aug = mx.image.CastAug()
    cla_resize_aug = mx.image.ForceResizeAug(size=[28, 28])
    cla_augmenters = [cla_cast_aug, cla_resize_aug]

    for aug in cla_augmenters:
        data = aug(data)
    data = mx.nd.transpose(data, axes=(2, 0, 1))
    data = mx.nd.expand_dims(data, axis=0)
    data = mx.io.DataBatch([data])
    return data

def get_output(model, data):
    model.forward(data)
    cla_prob = model.get_outputs()[0][0].asnumpy()
    cla_label = np.argmax(cla_prob)
    return cla_label

if __name__ == '__main__':
    model_prefix = "output/LeNet"
    index = 10
    context = mx.gpu(0)
    data_shapes = [('data', (1,1,28,28))]
    label_shapes = [('softmax_label', (1,))]
    model = load_model(model_prefix, index, context, data_shapes, label_shapes)

    data_path = "test_image/test1.png"
    data = load_data(data_path)

    cla_label = get_output(model, data)
    print("Predict result: {}".format(cla_label))
