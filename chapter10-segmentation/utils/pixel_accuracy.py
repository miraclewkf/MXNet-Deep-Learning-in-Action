import mxnet as mx
import numpy as np

class PixelAccuracy(mx.metric.EvalMetric):
    def __init__(self, name='Pixel Acc'):
        super(PixelAccuracy, self).__init__(name)
        self.name = name
        self.reset()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            pred = np.argmax(pred.asnumpy(), axis=1).astype('int32') + 1
            label = label.asnumpy().astype('int32') + 1

            pixel_valid = np.sum(label > 0)
            pixel_correct = np.sum((pred == label) * (label > 0))

            self.sum_metric += pixel_correct
            self.num_inst += pixel_valid

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)


