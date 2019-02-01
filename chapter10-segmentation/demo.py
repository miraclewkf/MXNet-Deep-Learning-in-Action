import mxnet as mx
from PIL import Image
import numpy as np

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

def get_data(img_path, rgb_mean):
    img = Image.open(img_path)
    data = mx.nd.array(img, dtype='float32')
    data = data - mx.nd.array(rgb_mean).reshape((1, 1, 3))
    data = mx.nd.transpose(data, axes=(2, 0, 1))
    data = mx.nd.expand_dims(data, axis=0)
    data_shapes = [(('data'), data.shape)]
    label_shapes = [(('softmax_label'), (1,) + data.shape[2:])]
    data = mx.io.DataBatch([data])
    return data, data_shapes, label_shapes

def load_model(model_prefix, index, context, data_shapes, label_shapes):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 
                                                           index)
    model = mx.mod.Module(symbol=sym, context=context)
    model.bind(for_training=False,
               data_shapes=data_shapes,
               label_shapes=label_shapes)
    model.set_params(arg_params=arg_params, 
                     aux_params=aux_params,
                     allow_missing=True)
    return model

def get_output(model, data, result_save):
    model.forward(data)
    cla_prob = model.get_outputs()[0].asnumpy()
    colormap = mx.nd.array(VOC_COLORMAP, dtype='uint8')
    out_mask = colormap[np.uint8(np.squeeze(cla_prob.argmax(axis=1)))]
    out_mask = Image.fromarray(out_mask.asnumpy())
    out_mask.save(result_save)

if __name__ == '__main__':
    model_prefix = "output/FCN32s/fcn32s"
    index = 50
    context = mx.gpu(0)

    img_path = 'demo_img/2007_003910.jpg'
    rgb_mean = (123.68, 116.779, 103.939)
    data, data_shapes, label_shapes = get_data(img_path, rgb_mean)

    model = load_model(model_prefix, index, context, data_shapes, label_shapes)
    result_save = img_path[:-4] + "_seg" + ".png"
    get_output(model, data, result_save)
