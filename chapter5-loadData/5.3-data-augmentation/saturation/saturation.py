import matplotlib.pyplot as plt
import mxnet as mx

if __name__ == '__main__':
    image = 'ILSVRC2012_val_00000008.JPEG'
    image_name = image.split(".")[0]
    image_string = open('../image/{}'.format(image), 'rb').read()
    data = mx.image.imdecode(image_string, flag=1)
    plt.imshow(data.asnumpy())
    plt.savefig('{}_original.png'.format(image_name))

    cast = mx.image.CastAug()
    data = cast(data)
    saturation = mx.image.SaturationJitterAug(saturation=0.3)
    saturation_data = saturation(data)
    saturation_data = mx.nd.Cast(saturation_data, dtype='uint8')
    plt.imshow(saturation_data.asnumpy())
    plt.savefig('{}_saturation.png'.format(image_name))