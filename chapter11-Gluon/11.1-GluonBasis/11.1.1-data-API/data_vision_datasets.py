from mxnet.gluon.data import vision
import matplotlib.pyplot as plt

label_text = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

cifar10 = vision.datasets.CIFAR10(root='data/',train=True)
fig = plt.figure()
img_num = 6
for i in range(img_num):
    fig.add_subplot(2,3,i+1)
    img, label = cifar10[i]
    plt.imshow(img.asnumpy())
    plt.title(label_text[label])
plt.savefig('cifar_10_img_sample.png')
