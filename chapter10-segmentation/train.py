import argparse
import os
import numpy as np
import logging
from data.voc import VocSegData
from symbol.FCN import *
from utils.mean_IoU import MeanIoU
from utils.pixel_accuracy import PixelAccuracy
from utils.pixel_ce import PixelCrossEntropy

def init_deconv(args, fcnxs, fcnxs_args):
    arr_name = fcnxs.list_arguments()
    shape_dic = {}
    if args.model == 'fcn32s':
        bigscore_kernel_size = 64
        init_layer = ["bigscore_weight"]
    elif args.model == 'fcn16s':
        bigscore_kernel_size = 32
        init_layer = ["bigscore_weight", "score2_weight"]
    else:
        bigscore_kernel_size = 16
        init_layer = ["bigscore_weight", "score4_weight"]
    shape_dic["bigscore_weight"] = {"in_channels": 21, "out_channels": 21,
                                    "kernel_size": bigscore_kernel_size}
    shape_dic["score2_weight"] = {"in_channels": 21, "out_channels": 21,
                                  "kernel_size": 4}
    shape_dic["score4_weight"] = {"in_channels": 21, "out_channels": 21,
                                  "kernel_size": 4}
    for arr in arr_name:
        if arr in init_layer:
            kernel_size = shape_dic[arr]["kernel_size"]
            in_channels = shape_dic[arr]["in_channels"]
            out_channels = shape_dic[arr]["out_channels"]
            factor = (kernel_size + 1) // 2
            if kernel_size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:kernel_size, :kernel_size]
            filt = (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
            weight = np.zeros(shape=(in_channels, out_channels,
                                     kernel_size, kernel_size),
                              dtype='float32')
            weight[range(in_channels), range(out_channels), :, :] = filt
            fcnxs_args[arr] = mx.nd.array(weight, dtype='float32')
    return fcnxs_args

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='batch size for train', default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--num-classes', type=int, help="number of classes", default=21)
    parser.add_argument('--data-dir', type=str, help="path for data", default='data/VOC2012')
    parser.add_argument('--model', type=str, help="type of FCN", default='fcn32s')
    parser.add_argument('--prefix', type=str, help="pretrain model", default='model/VGG_FC_ILSVRC_16_layers')
    parser.add_argument('--pretrain-epoch', type=int, help="index of pretrain model", default=74)
    parser.add_argument('--begin-epoch', type=int, help="begin epoch fro training", default=0)
    parser.add_argument('--num-epoch', type=int, help="number of training epoch", default=50)
    parser.add_argument('--rgb-mean', type=tuple, help="tuple of RGB mean", default=(123.68, 116.779, 103.939))
    parser.add_argument('--save-result', type=str, default='output/FCN32s/')
    parser.add_argument('--num-examples', type=int, default=1464)
    parser.add_argument('--step', type=str, default='40')
    parser.add_argument('--factor', type=int, default=0.2)
    args = parser.parse_args()
    return args

def multi_factor_scheduler(args, epoch_size):
    step = range(args.step, args.num_epoch, args.step)
    step_bs = [epoch_size * (x - args.begin_epoch) for x in step
             if x - args.begin_epoch > 0]
    if step_bs:
        return mx.lr_scheduler.MultiFactorScheduler(step=step_bs,
                                                    factor=args.factor)
    return None

def main():
    args = parse_arguments()
    if not os.path.exists(args.save_result):
        os.makedirs(args.save_result)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.save_result + 'train.log')
    logger.addHandler(file_handler)
    logger.info(args)

    if args.gpus == '':
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    if args.model == "fcn32s":
        fcn = symbol_fcn32s(num_classes=args.num_classes)
    elif args.model == "fcn16s":
        fcn = symbol_fcn16s(num_classes=args.num_classes)
    elif args.model == "fcn8s":
        fcn = symbol_fcn8s(num_classes=args.num_classes)
    else:
        print("Please set model as fcn32s or fcn16s or fcn8s.")
    _, arg_params, aux_params = mx.model.load_checkpoint(args.prefix,
                                                         args.pretrain_epoch)
    arg_params = init_deconv(args, fcn, arg_params)

    train_data = VocSegData(data_dir=args.data_dir,
                            lst_name="train.lst",
                            rgb_mean=args.rgb_mean)
    val_data = VocSegData(data_dir=args.data_dir,
                          lst_name="val.lst",
                          rgb_mean=args.rgb_mean)

    epoch_size = max(int(args.num_examples / args.batch_size), 1)
    step = [int(step_i.strip()) for step_i in args.step.split(",")]
    step_bs = [epoch_size * (x - args.begin_epoch) for x in step
               if x - args.begin_epoch > 0]
    if step_bs:
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=step_bs,
                                                            factor=args.factor)
    else:
        lr_scheduler = None

    optimizer_params = {'learning_rate': args.lr,
                        'momentum': args.mom,
                        'wd': args.wd,
                        'lr_scheduler': lr_scheduler}

    initializer = mx.init.Xavier(rnd_type='gaussian',
                                 factor_type="in",
                                 magnitude=2)
    model = mx.mod.Module(context=ctx, symbol=fcn)

    batch_callback = mx.callback.Speedometer(args.batch_size, 500)
    epoch_callback = mx.callback.do_checkpoint(args.save_result + args.model,
                                               period=2)
    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(PixelCrossEntropy())
    val_metric = mx.metric.CompositeEvalMetric()
    val_metric.add(PixelAccuracy())
    val_metric.add(MeanIoU())

    model.fit(train_data=train_data,
              eval_data=val_data,
              begin_epoch=args.begin_epoch,
              num_epoch=args.num_epoch,
              eval_metric=eval_metric,
              validation_metric=val_metric,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              arg_params=arg_params,
              aux_params=aux_params,
              initializer=initializer,
              allow_missing=True,
              batch_end_callback=batch_callback,
              epoch_end_callback=epoch_callback)

if __name__ == '__main__':
    main()