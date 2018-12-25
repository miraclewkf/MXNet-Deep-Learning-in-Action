import matplotlib.pyplot as plt
import mxnet as mx

if __name__ == '__main__':
    image = 'ILSVRC2012_val_00000002.JPEG'
    image_name = image.split(".")[0]
    image_string = open('../image/{}'.format(image), 'rb').read()
    data = mx.image.imdecode(image_string, flag=1)
    print("Shape of data:{}".format(data.shape))
    plt.imshow(data.asnumpy())
    plt.savefig('{}_original.png'.format(image_name))

    # class
    shorterResize = mx.image.ResizeAug(size=224)
    shorterResize_data = shorterResize(data)
    print("Shape of data:{}".format(shorterResize_data.shape))
    plt.imshow(shorterResize_data.asnumpy())
    plt.savefig('{}_shorterResize.png'.format(image_name))

    shorterResize = mx.image.ResizeAug(size=1000)
    shorterResize_data = shorterResize(data)
    print("Shape of data:{}".format(shorterResize_data.shape))
    plt.imshow(shorterResize_data.asnumpy())
    plt.savefig('{}_shorterResize_bigsize.png'.format(image_name))

    forceResize = mx.image.ForceResizeAug(size=(224,224))
    forceResize_data = forceResize(data)
    print("Shape of data:{}".format(forceResize_data.shape))
    plt.imshow(forceResize_data.asnumpy())
    plt.savefig('{}_forceResize.png'.format(image_name))

    forceResize = mx.image.ForceResizeAug(size=(200, 300))
    forceResize_data = forceResize(data)
    print("Shape of data:{}".format(forceResize_data.shape))
    plt.imshow(forceResize_data.asnumpy())
    plt.savefig('{}_forceResize_diff.png'.format(image_name))

    '''
    # function
    shorter_resize_data = mx.image.resize_short(data, 224)
    print("Shape of data:{}".format(shorter_resize_data.shape))
    plt.imshow(shorter_resize_data.asnumpy())
    plt.savefig('{}_shorter_resize.png'.format(image))

    force_resize_data = mx.nd._internal._cvimresize(data, 224, 224)
    print("Shape of data:{}".format(force_resize_data.shape))
    plt.imshow(force_resize_data.asnumpy())
    plt.savefig('{}_force_resize.png'.format(image_name))
    '''