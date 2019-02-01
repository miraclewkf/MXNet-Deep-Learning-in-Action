import mxnet as mx
import argparse
from symbol.get_ssd import get_ssd
from tools.custom_metric import MultiBoxMetric
from eval_metric_07 import VOC07MApMetric
import logging
import os
from data.dataiter import CustomDataIter
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=20)
    parser.add_argument('--num-examples', type=int, default=16551)
    parser.add_argument('--begin-epoch', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=240)
    parser.add_argument('--step', type=str, default='160,200')
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--frequent', type=int, default=20)
    parser.add_argument('--save-result', type=str, default='output/ssd_vgg/')
    parser.add_argument('--save-name', type=str, default='ssd')
    parser.add_argument('--class-names', type=str, default='aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor')
    parser.add_argument('--train-rec', type=str, default='data/VOCdevkit/VOC/ImageSets/Main/trainval.rec')
    parser.add_argument('--train-idx', type=str, default='data/VOCdevkit/VOC/ImageSets/Main/trainval.idx')
    parser.add_argument('--val-rec', type=str, default='data/VOCdevkit/VOC/ImageSets/Main/test.rec')
    parser.add_argument('--backbone-prefix', type=str, default='model/vgg16_reduced')
    parser.add_argument('--backbone-epoch', type=int, default=1)
    parser.add_argument('--freeze-layers', type=str, default="^(conv1_|conv2_).*")
    parser.add_argument('--data-shape', type=int, default=300)
    parser.add_argument('--label-pad-width', type=int, default=420)
    parser.add_argument('--label-name', type=str, default='label')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    if not os.path.exists(args.save_result):
        os.makedirs(args.save_result)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.save_result + '/train.log')
    logger.addHandler(file_handler)
    logger.info(args)

    if args.gpus == '':
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    train_data = CustomDataIter(args, is_trainData=True)
    val_data = CustomDataIter(args)

    ssd_symbol = get_ssd(num_classes=args.num_classes)
    vgg,arg_params,aux_params = mx.model.load_checkpoint(args.backbone_prefix,
                                                         args.backbone_epoch)

    if args.freeze_layers.strip():
        re_prog = re.compile(args.freeze_layers)
        fixed_param_names = [name for name in vgg.list_arguments() if
                             re_prog.match(name)]
    else:
        fixed_param_names = None

    mod = mx.mod.Module(symbol=ssd_symbol, label_names=(args.label_name,),
                        context=ctx, fixed_param_names=fixed_param_names)

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
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0/len(ctx) if len(ctx)>0 else 1.0}
    initializer = mx.init.Xavier(rnd_type='gaussian',
                                 factor_type='out',
                                 magnitude=2)

    class_names = [name_i for name_i in args.class_names.split(",")]
    VOC07_metric = VOC07MApMetric(ovp_thresh=0.5, use_difficult=False,
                                  class_names=class_names, pred_idx=3)
    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(MultiBoxMetric(name=['CrossEntropy Loss',
                                         'SmoothL1 Loss']))

    batch_callback = mx.callback.Speedometer(batch_size=args.batch_size,
                                             frequent=args.frequent)
    checkpoint_prefix = args.save_result+args.save_name
    epoch_callback = mx.callback.do_checkpoint(prefix=checkpoint_prefix,
                                               period=5)
    mod.fit(train_data=train_data,
            eval_data=val_data,
            eval_metric=eval_metric,
            validation_metric=VOC07_metric,
            epoch_end_callback=epoch_callback,
            batch_end_callback=batch_callback,
            optimizer='sgd',
            optimizer_params=optimizer_params,
            initializer=initializer,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            num_epoch=args.num_epoch)

if __name__ == '__main__':
    main()