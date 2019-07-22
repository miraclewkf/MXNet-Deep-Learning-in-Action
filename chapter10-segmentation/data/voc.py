import mxnet as mx
from PIL import Image
import os
import numpy as np

class VocSegData(mx.io.DataIter):
    def __init__(self, data_dir, lst_name, rgb_mean,
                 batch_size=1):
        super(VocSegData, self).__init__()
        self.data_dir = data_dir
        self.lst_name = lst_name
        self.rgb_mean = rgb_mean
        self.batch_size = batch_size
        self.data_name = 'data'
        self.label_name = 'softmax_label'
        self.cursor = -self.batch_size
        lst_path = os.path.join(data_dir, lst_name)
        self.num_data = len(open(lst_path, 'r').readlines())
        self.data_file = open(lst_path, 'r')
        self.data, self.label = self._read_image()
        self.reset()

    def _read_image(self):
        sample = self.data_file.readline().strip()
        index, img_path, label_path = sample.split("\t")
        img = Image.open(os.path.join(self.data_dir, img_path))
        mask = Image.open(os.path.join(self.data_dir, label_path))

        data = mx.nd.array(img, dtype='float32')
        data = data - mx.nd.array(self.rgb_mean).reshape((1, 1, 3))
        data = mx.nd.transpose(data, axes=(2, 0, 1))
        data = mx.nd.expand_dims(data, axis=0)

        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        target = mx.nd.expand_dims(mx.nd.array(target), axis=0)
        return [list([(self.data_name, data)]),
                list([(self.label_name, target)])]

    @property
    def provide_data(self):
        return [mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])),
                               v.dtype)
                for k, v in self.data]

    @property
    def provide_label(self):
        return [mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])),
                               v.dtype)
                for k, v in self.label]

    def reset(self):
        self.cursor = -self.batch_size
        self.data_file.close()
        self.data_file = open(os.path.join(self.data_dir, self.lst_name), 'r')

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            data, label = self._getdata()
            return mx.io.DataBatch(data=data,
                                   label=label,
                                   pad=self.getpad(),
                                   index=None,
                                   provide_data=self.provide_data,
                                   provide_label=self.provide_label)
        else:
            raise StopIteration

    def _getdata(self):
        if self.cursor+self.batch_size <= self.num_data:
            self.data, self.label = self._read_image()
            return [x[1] for x in self.data], [x[1] for x in self.label]

    def gepad(self):
        return 0
