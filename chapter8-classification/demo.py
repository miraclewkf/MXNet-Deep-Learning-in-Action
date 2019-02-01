import mxnet as mx
import numpy as np

def load_model(model_prefix, index, context, data_shapes, label_shapes):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, index)
    model = mx.mod.Module(symbol=sym, context=context)
    model.bind(for_training=False,
               data_shapes=data_shapes,
               label_shapes=label_shapes)
    model.set_params(arg_params=arg_params,
                     aux_params=aux_params,
                     allow_missing=True)
    return model

def load_data(data_path):
    data = mx.image.imread(data_path)
    cast_aug = mx.image.CastAug()
    resize_aug = mx.image.ForceResizeAug(size=[224, 224])
    norm_aug = mx.image.ColorNormalizeAug(mx.nd.array([123, 117, 104]),
                                          mx.nd.array([1, 1, 1]))
    cla_augmenters = [cast_aug, resize_aug, norm_aug]

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

def main():
    label_map = {0: "cat", 1: "dog"}
    model_prefix = "output/resnet-18/resnet-18"
    index = 10
    context = mx.gpu(0)
    data_shapes = [('data', (1, 3, 224, 224))]
    label_shapes = [('softmax_label', (1,))]
    model = load_model(model_prefix, index, context, data_shapes, label_shapes)

    data_path = "data/demo_img1.jpg"
    data = load_data(data_path)

    cla_label = get_output(model, data)
    print("Predict result: {}".format(label_map.get(cla_label)))

if __name__ == '__main__':
    main()
