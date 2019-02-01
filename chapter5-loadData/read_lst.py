import mxnet as mx
import matplotlib.pyplot as plt

train_data = mx.image.ImageIter(batch_size=32,
                                data_shape=(3,224,224),
                                path_imglist='data/train.lst',
                                path_root='data/train',
                                shuffle=True)
val_data = mx.image.ImageIter(batch_size=32,
                              data_shape=(3,224,224),
                              path_imglist='data/val.lst',
                              path_root='data/val')

train_data.reset()
data_batch = train_data.next()
data = data_batch.data[0]
plt.figure()
for i in range(4):
    save_image = data[i].astype('uint8').asnumpy().transpose((1,2,0))
    plt.subplot(1,4,i+1)
    plt.imshow(save_image)
plt.savefig('image_sample.jpg')

train_data = mx.image.ImageIter(batch_size=32,
                                data_shape=(3, 224, 224),
                                path_imglist='data/train.lst',
                                path_root='data/train',
                                shuffle=True,
                                resize=256,
                                rand_mirror=True)



