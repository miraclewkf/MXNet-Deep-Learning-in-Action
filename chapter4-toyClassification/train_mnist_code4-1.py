import mxnet as mx
import argparse
import numpy as np
import gzip
import struct
import logging

def get_network(num_classes):
    """
    LeNet
    """
    data = mx.sym.Variable("data")
    conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=6, 
                               name="conv1")
    relu1 = mx.sym.Activation(data=conv1, act_type="relu", name="relu1")
    pool1 = mx.sym.Pooling(data=relu1, kernel=(2,2), stride=(2,2), 
                           pool_type="max", name="pool1")

    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=16, 
                               name="conv2")
    relu2 = mx.sym.Activation(data=conv2, act_type="relu", name="relu2")
    pool2 = mx.sym.Pooling(data=relu2, kernel=(2, 2), stride=(2, 2), 
                           pool_type="max", name="pool2")

    fc1 = mx.sym.FullyConnected(data=pool2, num_hidden=120, name="fc1")
    relu3 = mx.sym.Activation(data=fc1, act_type="relu", name="relu3")

    fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=84, name="fc2")
    relu4 = mx.sym.Activation(data=fc2, act_type="relu", name="relu4")

    fc3 = mx.sym.FullyConnected(data=relu4, num_hidden=num_classes, name="fc3")
    sym = mx.sym.SoftmaxOutput(data=fc3, name="softmax")
    return sym

def get_args():
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--save-result', type=str, default='output/')
    parser.add_argument('--save-name', type=str, default='LeNet')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.gpus:
        context = [mx.gpu(int(index)) for index in 
                   args.gpus.strip().split(",")]
    else:
        context = mx.cpu()

    # get data
    train_data = mx.io.MNISTIter(
        image='train-images.idx3-ubyte',
        label='train-labels.idx1-ubyte',
        batch_size=args.batch_size,
        shuffle=1)
    val_data = mx.io.MNISTIter(
        image='t10k-images.idx3-ubyte',
        label='t10k-labels.idx1-ubyte',
        batch_size=args.batch_size,
        shuffle=0)

    # get network(symbol)
    sym = get_network(num_classes=args.num_classes)

    optimizer_params = {'learning_rate': args.lr}
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", 
                                 magnitude=2)

    mod = mx.mod.Module(symbol=sym, context=context)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler('output/train.log')
    logger.addHandler(file_handler)
    logger.info(args)

    checkpoint = mx.callback.do_checkpoint(prefix=args.save_result + 
                                           args.save_name)
    batch_callback = mx.callback.Speedometer(args.batch_size, 1000)
    mod.fit(train_data=train_data,
            eval_data=val_data,
            eval_metric = 'acc',
            optimizer_params=optimizer_params,
            optimizer='sgd',
            batch_end_callback=batch_callback,
            initializer=initializer,
            num_epoch = args.num_epoch,
            epoch_end_callback=checkpoint)
