import mxnet as mx
from PIL import Image
import os

class VocSegData(mx.io.DataIter):
    def __init__(self, data_dir, lst_name, rgb_mean,
                 batch_size=1, data_name='data', label_name='softmax_label'):
        super(VocSegData, self).__init__()
        self.data_dir = data_dir
        self.lst_name = lst_name
        self.rgb_mean = rgb_mean
        self.batch_size = batch_size
        self.data_name = data_name
        self.label_name = label_name
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
        data = mx.nd.array(img, dtype='float32')
        data = data - mx.nd.array(self.rgb_mean).reshape((1, 1, 3))
        data = mx.nd.transpose(data, axes=(2, 0, 1))
        data = mx.nd.expand_dims(data, axis=0)

        label = Image.open(os.path.join(self.data_dir, label_path))
        label = mx.nd.array(label)
        label = mx.nd.expand_dims(label, axis=0)
        return list([(self.data_name, data)]), list([(self.label_name, label)])

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
