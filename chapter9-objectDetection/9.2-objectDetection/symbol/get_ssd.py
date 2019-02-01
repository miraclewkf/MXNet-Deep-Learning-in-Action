from .vggnet import *

def config():
    config_dict = {}
    config_dict['from_layers'] = ['relu4_3', 'relu7', '', '', '', '']
    config_dict['num_filters'] = [512, -1, 512, 256, 256, 256]
    config_dict['strides'] = [-1, -1, 2, 2, 1, 1]
    config_dict['pads'] = [-1, -1, 1, 1, 0, 0]
    config_dict['normalization'] = [20, -1, -1, -1, -1, -1]
    config_dict['sizes'] = [[0.1, 0.141], [0.2, 0.272], [0.37, 0.447],
                            [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    config_dict['ratios'] = [[1, 2, 0.5], [1, 2, 0.5, 3, 1.0/3],
                             [1, 2, 0.5, 3, 1.0/3], [1, 2, 0.5, 3, 1.0/3],
                             [1, 2, 0.5], [1, 2, 0.5]]
    config_dict['steps'] = [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
    return config_dict

def add_extras(backbone, config_dict):
    layers = []
    body = backbone.get_internals()
    for i, from_layer in enumerate(config_dict['from_layers']):
        if from_layer is '':
            layer = layers[-1]
            num_filters = config_dict['num_filters'][i]
            s = config_dict['strides'][i]
            p = config_dict['pads'][i]
            conv_1x1 = mx.sym.Convolution(data=layer, kernel=(1,1),
                                          num_filter=num_filters // 2,
                                          pad=(0,0), stride=(1,1),
                                          name="conv{}_1".format(i+6))
            relu_1 = mx.sym.Activation(data=conv_1x1, act_type='relu',
                                       name="relu{}_1".format(i+6))
            conv_3x3 = mx.sym.Convolution(data=relu_1, kernel=(3,3),
                                          num_filter=num_filters,
                                          pad=(p,p), stride=(s,s),
                                          name="conv{}_2".format(i+6))
            relu_2 = mx.sym.Activation(data=conv_3x3, act_type='relu',
                                       name="relu{}_2".format(i+6))
            layers.append(relu_2)
        else:
            layers.append(body[from_layer + '_output'])
    return layers

def create_predictor(from_layers, config_dict, num_classes):
    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1

    for i, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        if config_dict['normalization'][i] > 0:
            num_filters = config_dict['num_filters'][i]
            init = mx.init.Constant(config_dict['normalization'][i])
            L2_normal = mx.sym.L2Normalization(data=from_layer, mode="channel",
                                              name="{}_norm".format(from_name))
            scale = mx.sym.Variable(name="{}_scale".format(from_name),
                                    shape=(1, num_filters, 1, 1),
                                    init=init, attr={'__wd_mult__': '0.1'})
            from_layer = mx.sym.broadcast_mul(lhs=scale, rhs=L2_normal)

        anchor_size = config_dict['sizes'][i]
        anchor_ratio = config_dict['ratios'][i]
        num_anchors = len(anchor_size) - 1 + len(anchor_ratio)

        # regression layer
        num_loc_pred = num_anchors * 4
        weight = mx.sym.Variable(name="{}_loc_pred_conv_weight".format(from_name),
                                 init=mx.init.Xavier(magnitude=2))
        loc_pred = mx.sym.Convolution(data=from_layer, kernel=(3,3),
                                      weight=weight, pad=(1,1), 
                                      num_filter=num_loc_pred,
                                      name="{}_loc_pred_conv".format(
                                      from_name))
        loc_pred = mx.sym.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.sym.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        # classification part
        num_cls_pred = num_anchors * num_classes
        weight = mx.sym.Variable(name="{}_cls_pred_conv_weight".format(from_name),
                                 init=mx.init.Xavier(magnitude=2))
        cls_pred = mx.sym.Convolution(data=from_layer, kernel=(3,3),
                                      weight=weight, pad=(1,1), 
                                      num_filter=num_cls_pred,
                                      name="{}_cls_pred_conv".format(
                                      from_name))
        cls_pred = mx.sym.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.sym.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)

        # anchor part
        anchor_step = config_dict['steps'][i]
        anchors = mx.sym.contrib.MultiBoxPrior(from_layer, sizes=anchor_size,
                                               ratios=anchor_ratio, clip=False,
                                               steps=(anchor_step,anchor_step),
                                               name="{}_anchors".format(from_name))
        anchors = mx.sym.Flatten(data=anchors)
        anchor_layers.append(anchors)
    loc_preds = mx.sym.concat(*loc_pred_layers, name="multibox_loc_preds")
    cls_preds = mx.sym.concat(*cls_pred_layers)
    cls_preds = mx.sym.reshape(data=cls_preds, shape=(0,-1,num_classes))
    cls_preds = mx.sym.transpose(cls_preds, axes=(0,2,1), name="multibox_cls_preds")
    anchors = mx.sym.concat(*anchor_layers)
    anchors = mx.sym.reshape(data=anchors, shape=(0,-1,4), name="anchors")
    return loc_preds, cls_preds, anchors

def create_multi_loss(label, loc_preds, cls_preds, anchors):
    loc_target,loc_target_mask,cls_target = mx.sym.contrib.MultiBoxTarget(
        anchor=anchors,
        label=label,
        cls_pred=cls_preds,
        overlap_threshold=0.5,
        ignore_label=-1,
        negative_mining_ratio=3,
        negative_mining_thresh=0.5,
        minimum_negative_samples=0,
        variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")

    cls_prob = mx.sym.SoftmaxOutput(data=cls_preds, label=cls_target,
                                    ignore_label=-1, use_ignore=True,
                                    multi_output=True,
                                    normalization='valid',
                                    name="cls_prob")
    loc_loss_ = mx.sym.smooth_l1(data=loc_target_mask*(loc_preds-loc_target),
                                 scalar=1.0,
                                 name="loc_loss_")
    loc_loss = mx.sym.MakeLoss(loc_loss_, normalization='valid',
                               name="loc_loss")

    cls_label = mx.sym.MakeLoss(data=cls_target, grad_scale=0,
                                name="cls_label")
    det = mx.sym.contrib.MultiBoxDetection(cls_prob=cls_prob,
                                           loc_pred=loc_preds,
                                           anchor=anchors,
                                           nms_threshold=0.45,
                                           force_suppress=False,
                                           nms_topk=400,
                                           variances=(0.1,0.1,0.2,0.2),
                                           name="detection")
    det = mx.sym.MakeLoss(data=det, grad_scale=0, name="det_out")
    output = mx.sym.Group([cls_prob, loc_loss, cls_label, det])
    return output

def get_ssd(num_classes):
    config_dict = config()
    backbone = VGGNet()
    from_layers = add_extras(backbone=backbone,
                             config_dict=config_dict)
    loc_preds, cls_preds, anchors = create_predictor(from_layers=from_layers,
                                                     config_dict=config_dict,
                                                     num_classes=num_classes)
    label = mx.sym.Variable('label')
    ssd_symbol = create_multi_loss(label=label, loc_preds=loc_preds,
                                   cls_preds=cls_preds, anchors=anchors)
    return ssd_symbol
