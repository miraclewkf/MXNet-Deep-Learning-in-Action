from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms, datasets

transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=0.13, std=0.31)])
cifar10_train = datasets.CIFAR10(root='data/',
                                 train=True).transform_first(transformer)
train_data = DataLoader(dataset=cifar10_train,
                        batch_size=8,
                        shuffle=True,
                        num_workers=0)
for data, label in train_data:
    print("Shape of data: {}".format(data.shape))
    print("Shape of label: {}".format(label.shape))
    break
