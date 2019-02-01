import mxnet as mx
import numpy as np

class PixelCrossEntropy(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12, name='pixel-cross-entropy'):
        super(PixelCrossEntropy, self).__init__(name, eps=eps)
        self.eps = eps
        self.name = name
        self.reset()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            num_class = pred.shape[1]
            label = label.asnumpy()
            pred = pred.transpose((0, 2, 3, 1)).reshape((-1, num_class))
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            invalid_label_idx = np.where(label == -1)
            # equal to set loss=0
            prob[invalid_label_idx] = 1
            valid_label_num = label.shape[0] - len(invalid_label_idx[0])
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += valid_label_num