import mxnet as mx
from symbol.get_ssd import get_ssd
import numpy as np
import random, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model(prefix, index, context):
    batch_size = 1
    data_width, data_height = 300, 300
    data_names = ['data']
    data_shapes = [('data', (batch_size, 3, data_height, data_width))]
    label_names = ['label']
    label_shapes = [('label', (batch_size, 3, 6))]
    body, arg_params, aux_params = mx.model.load_checkpoint(prefix, index)
    symbol = get_ssd(num_classes=20)
    model = mx.mod.Module(symbol=symbol, context=context,
                          data_names=data_names,
                          label_names=label_names)
    model.bind(for_training=False,
               data_shapes=data_shapes,
               label_shapes=label_shapes)
    model.set_params(arg_params=arg_params, aux_params=aux_params,
                     allow_missing=True)
    return model

def transform(data, augmenters):
    for aug in augmenters:
        data = aug(data)
    return data

def plot_pred(det, data, img_i):
    height = data.shape[0]
    width = data.shape[1]

    colors = dict()
    label_name = ["aeroplane","bicycle","bird","boat","bottle", \
                  "bus","car","cat","chair","cow","diningtable", \
                  "dog","horse","motorbike","person","pottedplant", \
                  "sheep","sofa","train","tvmonitor"]
    fig, ax = plt.subplots()
    plt.imshow(data.asnumpy())
    for i in range(det.shape[0]):
        cls_id = int(det[i, 0])
        score = det[i, 1]
        if score > 0.5:
            xmin = det[i, 2] * width
            ymin = det[i, 3] * height
            xmax = det[i, 4] * width
            ymax = det[i, 5] * height
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(),
                                  random.random())
            rect = patches.Rectangle(xy=(xmin,ymin), width=xmax-xmin,
                                     height=ymax-ymin,
                                     edgecolor=colors[cls_id],
                                     facecolor='None',
                                     linewidth=1.5)
            plt.text(xmin, ymin, label_name[cls_id] + " " + str(score),
                     bbox=dict(facecolor=colors[cls_id], alpha=0.5))
            ax.add_patch(rect)
    plt.axis('off')
    plt.savefig("detection_result/{}".format(img_i))

def main():
    model = load_model(prefix="output/ssd_vgg/ssd",
                       index=225,
                       context=mx.gpu(0))

    cast_aug = mx.image.CastAug()
    resize_aug = mx.image.ForceResizeAug(size=[300, 300])
    normalization = mx.image.ColorNormalizeAug(mx.nd.array([123, 117, 104]),
                                               mx.nd.array([1, 1, 1]))
    augmenters = [cast_aug, resize_aug, normalization]

    img_list = os.listdir('demo_img')
    for img_i in img_list:
        data = mx.image.imread('demo_img/'+img_i)
        det_data = transform(data, augmenters)
        det_data = mx.nd.transpose(det_data, axes=(2, 0, 1))
        det_data = mx.nd.expand_dims(det_data, axis=0)

        model.forward(mx.io.DataBatch((det_data,)))
        det_result = model.get_outputs()[3].asnumpy()
        det = det_result[np.where(det_result[:,:,0] >= 0)]
        plot_pred(det=det, data=data, img_i=img_i)

if __name__ == '__main__':
    main()
