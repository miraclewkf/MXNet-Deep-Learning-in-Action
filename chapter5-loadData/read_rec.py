import mxnet as mx
import matplotlib.pyplot as plt

train_data = mx.io.ImageRecordIter(batch_size=32,
                                   data_shape=(3,224,224),
                                   path_imgrec='data/train.rec',
                                   path_imgidx='data/train.idx',
                                   shuffle=True,
                                   resize=256,
                                   rand_mirror=True)

val_data = mx.io.ImageRecordIter(batch_size=32,
                                 data_shape=(3,224,224),
                                 path_imgrec='data/val.rec',
                                 path_imgidx='data/val.idx',
                                 resize=256)

train_data.reset()
data_batch = train_data.next()
data = data_batch.data[0]
plt.figure()
for i in range(4):
    save_image = data[i].astype('uint8').asnumpy().transpose((1,2,0))
    plt.subplot(1,4,i+1)
    plt.imshow(save_image)
plt.savefig('image_sample_rec.jpg')
