import argparse
from time import time
import logging
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from mxnet.gluon import model_zoo

def transform():
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(0.13, 0.31)])
    return transformer

def cifar10Data(batch_size, num_workers):
    cifar10_train = datasets.CIFAR10(root='data/',train=True
                                     ).transform_first(transform())
    train_data = gluon.data.DataLoader(cifar10_train,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)
    cifar10_val = datasets.CIFAR10(root='data/',train=False
                                   ).transform_first(transform())
    val_data = gluon.data.DataLoader(cifar10_val,
                                     batch_size=batch_size,
                                     num_workers=num_workers)
    return train_data, val_data

def acc(output, label):
    pre_label = output.argmax(axis=1)
    return (pre_label == label.astype('float32')).mean().asscalar()

def train(args, train_data, val_data, net):
    trainer = gluon.Trainer(params=net.collect_params(),
                            optimizer='sgd',
                            optimizer_params={'learning_rate': 0.05})
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    epoch_index = 0
    for epoch in range(args.num_epoch):
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        tic = time()
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            train_loss += loss.mean().asscalar()
            trainer.step(args.batch_size)
            train_acc += acc(output, label)
        for data, label in val_data:
            val_acc += acc(net(data), label)
        epoch_index += 1
        if epoch_index % args.save_step == 0:
            net.save_parameters("{}-{}.params".format(args.save_prefix, epoch))
            print("save model to {}-{}.params".format(args.save_prefix, epoch))
        print("Epoch {}: Loss {:.4f}, Train accuracy {:.4f}, \
               Val accuracy {:.4f}, Time {:.4f}sec".format(epoch,
                                                   train_loss/len(train_data),
                                                   train_acc/(len(train_data)),
                                                   val_acc/(len(val_data)),
                                                   time()-tic))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='batch size for train', default=256)
    parser.add_argument('--num-epoch', type=int, help="number of training epoch", default=10)
    parser.add_argument('--num-workers', type=int, help="number of workers for data reading", default=8)
    parser.add_argument('--save-prefix', type=str, help="path to save model", default="output/ResNet18")
    parser.add_argument('--save-step', type=int, help="step of epoch to save model", default=5)
    parser.add_argument('--use-hybrid', type=bool, help="use hybrid or not", default=False)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    train_data, val_data = cifar10Data(args.batch_size, args.num_workers)
    net = model_zoo.vision.resnet18_v1()
    net.output = nn.Dense(10)
    net.initialize()
    if args.use_hybrid == True:
        net.hybridize()
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler('output/train.log')
    logger.addHandler(file_handler)
    logger.info(args)

    train(args, train_data, val_data, net)

if __name__ == '__main__':
    main()
