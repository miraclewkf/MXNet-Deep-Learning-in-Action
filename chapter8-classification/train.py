import argparse
import mxnet as mx
import os
import logging

def get_fine_tune_model(sym, num_classes, layer_name):
    all_layers = sym.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes,
                                   name='new_fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    return net

def multi_factor_scheduler(args, epoch_size):
    step = range(args.step, args.num_epoch, args.step)
    step_bs = [epoch_size * (x - args.begin_epoch) for x in step
             if x - args.begin_epoch > 0]
    if step_bs:
        return mx.lr_scheduler.MultiFactorScheduler(step=step_bs,
                                                    factor=args.factor)
    return None

def data_loader(args):
    data_shape_list = [int(item) for item in args.image_shape.split(",")]
    data_shape = tuple(data_shape_list)
    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_train_rec,
        path_imgidx=args.data_train_idx,
        label_width=1,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        data_name='data',
        label_name='softmax_label',
        data_shape=data_shape,
        batch_size=args.batch_size,
        rand_mirror=args.random_mirror,
        max_random_contrast=args.max_random_contrast,
        max_rotate_angle=args.max_rotate_angle,
        shuffle=True,
        resize=args.resize_train)

    val = mx.io.ImageRecordIter(
        path_imgrec=args.data_val_rec,
        path_imgidx=args.data_val_idx,
        label_width=1,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        data_name='data',
        label_name='softmax_label',
        data_shape=data_shape,
        batch_size=args.batch_size,
        rand_mirror=0,
        shuffle=False,
        resize=args.resize_val)
    return train,val

def train_model(args):
    train, val = data_loader(args)

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=args.model,
                                                           epoch=args.begin_epoch)
    new_sym = get_fine_tune_model(sym, args.num_classes, args.layer_name)

    epoch_size = max(int(args.num_examples / args.batch_size), 1)
    lr_scheduler = multi_factor_scheduler(args, epoch_size)

    optimizer_params = {'learning_rate': args.lr,
                        'momentum': args.mom,
                        'wd': args.wd,
                        'lr_scheduler': lr_scheduler}

    initializer = mx.init.Xavier(rnd_type='gaussian',
                                 factor_type="in",
                                 magnitude=2)

    if args.gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    if args.fix_pretrain_param:
        fixed_param_names = [layer_name for layer_name in
                             new_sym.list_arguments() if layer_name not in
                             ['new_fc_weight', 'new_fc_bias', 'data',
                              'softmax_label']]
    else:
        fixed_param_names = None

    model = mx.mod.Module(context=devs,
                          symbol=new_sym,
                          fixed_param_names=fixed_param_names)

    batch_callback = mx.callback.Speedometer(args.batch_size, args.period)
    epoch_callback = mx.callback.do_checkpoint(args.save_result + args.save_name)

    if args.from_scratch:
        arg_params = None
        aux_params = None

    model.fit(train_data=train,
              eval_data=val,
              begin_epoch=args.begin_epoch,
              num_epoch=args.num_epoch,
              eval_metric=['acc','ce'],
              optimizer='sgd',
              optimizer_params=optimizer_params,
              arg_params=arg_params,
              aux_params=aux_params,
              initializer=initializer,
              allow_missing=True,
              batch_end_callback=batch_callback,
              epoch_end_callback=epoch_callback)

def parse_arguments():
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model', type=str, default='model/resnet-18')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--begin-epoch', type=int, default=0)
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--resize-train', type=int, default=256)
    parser.add_argument('--resize-val', type=int, default=224)
    parser.add_argument('--data-train-rec', type=str, default='data/data_train.rec')
    parser.add_argument('--data-train-idx', type=str, default='data/data_train.idx')
    parser.add_argument('--data-val-rec', type=str, default='data/data_val.rec')
    parser.add_argument('--data-val-idx', type=str, default='data/data_val.idx')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--save-result', type=str, default='output/resnet-18/')
    parser.add_argument('--num-examples', type=int, default=22500)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--save-name', type=str, default='resnet-18')
    parser.add_argument('--random-mirror', type=int, default=1,
                        help='if or not randomly flip horizontally')
    parser.add_argument('--max-random-contrast', type=float, default=0.3,
                        help='Chanege the contrast with a value randomly chosen from [-max, max]')
    parser.add_argument('--max-rotate-angle', type=int, default=15,
                        help='Rotate by a random degree in [-v,v]')
    parser.add_argument('--layer-name', type=str, default='flatten0',
                        help='the layer name before fullyconnected layer')
    parser.add_argument('--factor', type=float, default=0.2, help='factor for learning rate decay')
    parser.add_argument('--step', type=int, default=5, help='step for learning rate decay')
    parser.add_argument('--from-scratch', type=bool, default=False,
                        help='Whether train from scratch')
    parser.add_argument('--fix-pretrain-param', type=bool, default=False,
                        help='Whether fix parameters of pretrain model')
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

    train_model(args=args)

if __name__ == '__main__':
    main()
