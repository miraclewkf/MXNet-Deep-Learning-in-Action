import matplotlib.pyplot as plt
import mxnet as mx

if __name__ == '__main__':
    image = 'ILSVRC2012_val_00000014.JPEG'
    image_name = image.split(".")[0]
    image_string = open('../image/{}'.format(image), 'rb').read()
    data = mx.image.imdecode(image_string, flag=1)
    print("Shape of data:{}".format(data.shape))
    plt.imshow(data.asnumpy())
    plt.savefig('{}_original.png'.format(image_name))

    mirror = mx.image.HorizontalFlipAug(p=0.5)
    mirror_data = mirror(data)
    plt.imshow(mirror_data.asnumpy())
    plt.savefig('{}_mirror.png'.format(image_name))