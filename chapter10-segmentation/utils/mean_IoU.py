import mxnet as mx
import numpy as np

class MeanIoU(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12, name='Mean IOU', num_classes=21):
        super(MeanIoU, self).__init__(name, eps=eps)
        self.eps = eps
        self.name = name
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            pred = np.argmax(pred.asnumpy(), 1).astype('int32') + 1
            label = label.asnumpy().astype('int32') + 1
            pred = pred * (label > 0).astype(pred.dtype)
            mx.metric.check_label_shapes(label, pred)

            mini = 1
            maxi = self.num_classes
            nbins = self.num_classes
            intersection = pred * (pred == label)
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection, bins=nbins, 
                                         range=(mini, maxi))
            area_pred, _ = np.histogram(pred, bins=nbins, range=(mini, maxi))
            area_lab, _ = np.histogram(label, bins=nbins, range=(mini, maxi))
            area_union = area_pred + area_lab - area_inter
        self.sum_metric += area_inter
        self.num_inst += area_union

    def get(self):
        if len(self.num_inst) == 0:
            return (self.name, float('nan'))
        else:
            IoU = 1.0 * self.sum_metric / (np.spacing(1) + self.num_inst)
            mIoU = IoU.mean()
            return (self.name, mIoU)
