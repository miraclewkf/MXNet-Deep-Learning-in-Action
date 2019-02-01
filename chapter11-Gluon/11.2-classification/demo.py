from mxnet.gluon import model_zoo
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms, datasets
import matplotlib.pyplot as plt

label_text = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

net = model_zoo.vision.resnet18_v1()
net.output = nn.Dense(10)
net.load_parameters(filename="output/ResNet18-4.params")

transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(0.13, 0.31)])

cifar10_val = datasets.CIFAR10(root='data/',train=False)
plt.figure()
for i in range(6):
    img, label = cifar10_val[i]
    data = transformer(img).expand_dims(axis=0)
    output = net.forward(data)
    pre_label = output.argmax(axis=1).astype("int32").asscalar()
    print("Predict label is: {}, Ground truth is: {}".format(
          label_text[pre_label], label_text[label]))

    plt.subplot(2,3,i+1)
    plt.axis('off')
    plt.imshow(img.asnumpy())
    plt.title("Predict: " + label_text[pre_label] + "\n" + "Truth: " + 
              label_text[label])
    plt.savefig("Prediction result.png")


