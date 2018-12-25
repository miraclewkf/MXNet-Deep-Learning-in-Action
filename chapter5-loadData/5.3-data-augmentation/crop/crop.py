import matplotlib.pyplot as plt
import mxnet as mx

if __name__ == '__main__':
    image = 'ILSVRC2012_val_00000009.JPEG'
    image_name = image.split(".")[0]
    image_string = open('../image/{}'.format(image), 'rb').read()
    data = mx.image.imdecode(image_string, flag=1)
    print("Shape of data:{}".format(data.shape))
    plt.imshow(data.asnumpy())
    plt.savefig('{}_original.png'.format(image_name))

    # class
    centerCrop = mx.image.CenterCropAug(size=(224,224))
    class_centerCrop_data = centerCrop(data)
    print("Shape of data:{}".format(class_centerCrop_data.shape))
    plt.imshow(class_centerCrop_data.asnumpy())
    plt.savefig('{}_centerCrop.png'.format(image_name))

    randomCrop = mx.image.RandomCropAug(size=(224,224))
    class_randomCrop_data = randomCrop(data)
    print("Shape of data:{}".format(class_randomCrop_data.shape))
    plt.imshow(class_randomCrop_data.asnumpy())
    plt.savefig('{}_randomCrop.png'.format(image_name))

    randomSizeCrop = mx.image.RandomSizedCropAug(size=(224,224), min_area=0.08, 
                                                 ratio=(3/4, 4/3))
    class_randomSizedCrop_data = randomSizeCrop(data)
    print("Shape of data:{}".format(class_randomSizedCrop_data.shape))
    plt.imshow(class_randomSizedCrop_data.asnumpy())
    plt.savefig('{}_randomSizedCrop.png'.format(image_name))

    '''
    # function
    function_centerCrop_data = mx.image.center_crop(data, (224, 224))
    print("Shape of data:{}".format(function_centerCrop_data[0].shape))
    plt.imshow(function_centerCrop_data[0].asnumpy())
    plt.savefig('crop/{}_function_centerCrop.png'.format(image))

    function_randomCrop_data = mx.image.random_crop(data, (224, 224))
    print("Shape of data:{}".format(function_randomCrop_data[0].shape))
    plt.imshow(function_randomCrop_data[0].asnumpy())
    plt.savefig('crop/{}_function_randomCrop.png'.format(image))

    function_randomSizedCrop_data = mx.image.random_size_crop(data, (224, 224), 0.08, (3 / 4, 4 / 3))
    print("Shape of data:{}".format(function_randomSizedCrop_data[0].shape))
    plt.imshow(function_randomSizedCrop_data[0].asnumpy())
    plt.savefig('crop/{}_function_randomSizedCrop.png'.format(image))
    '''