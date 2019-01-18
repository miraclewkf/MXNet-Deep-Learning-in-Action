import mxnet as mx

class Recall(mx.metric.EvalMetric):
    def __init__(self, name):
        super(Recall, self).__init__('Recall')
        self.name = name
        self.reset()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        for pred, label in zip(preds, labels):
            pred = mx.nd.argmax_channel(pred).asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            true_positives = 0
            false_negatives = 0
            for index in range(len(pred.flat)):
                if pred[index] == 0 and label[index] == 0:
                    true_positives += 1
                if pred[index] != 0 and label[index] == 0:
                    false_negatives += 1
            self.sum_metric += true_positives
            self.num_inst += (true_positives+false_negatives)

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)
