import mxnet as mx
import numpy as np

class MultiBoxMetric(mx.metric.EvalMetric):
    def __init__(self, name):
        super(MultiBoxMetric, self).__init__('MultiBoxMetric')
        self.name = name
        self.eps = 1e-18
        self.reset()

    def reset(self):
        self.num = 2
        self.num_inst = [0] * self.num
        self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()

        valid_count = np.sum(cls_label >= 0)
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]

        # CrossEntropy Loss
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count

        # SmoothL1 Loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

    def get(self):
        result = [sum / num if num != 0 else float('nan') for sum, num in zip(self.sum_metric, self.num_inst)]
        return (self.name, result)
