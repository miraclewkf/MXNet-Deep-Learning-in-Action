import mxnet as mx

class CustomDataIter(mx.io.DataIter):
    def __init__(self, args, is_trainData=False):
        self.args = args
        data_shape = (3, args.data_shape, args.data_shape)
        if is_trainData:
            self.data=mx.io.ImageDetRecordIter(
                path_imgrec=args.train_rec,
                batch_size=args.batch_size,
                data_shape=data_shape,
                mean_r=123.68,
                mean_g=116.779,
                mean_b=103.939,
                label_pad_width=420,
                random_hue_prob=0.5,
                max_random_hue=18,
                random_saturation_prob=0.5,
                max_random_saturation=32,
                random_illumination_prob=0.5,
                max_random_illumination=32,
                random_contrast_prob=0.5,
                max_random_contrast=0.5,
                rand_pad_prob=0.5,
                fill_value=127,
                max_pad_scale=4,
                rand_crop_prob=0.833333,
                max_crop_aspect_ratios=[2.0, 2.0, 2.0, 2.0, 2.0],
                max_crop_object_coverages=[1.0, 1.0, 1.0, 1.0, 1.0],
                max_crop_overlaps=[1.0, 1.0, 1.0, 1.0, 1.0],
                max_crop_sample_coverages=[1.0, 1.0, 1.0, 1.0, 1.0],
                max_crop_scales=[1.0, 1.0, 1.0, 1.0, 1.0],
                max_crop_trials=[25, 25, 25, 25, 25],
                min_crop_aspect_ratios=[0.5, 0.5, 0.5, 0.5, 0.5],
                min_crop_object_coverages=[0.0, 0.0, 0.0, 0.0, 0.0],
                min_crop_overlaps=[0.1, 0.3, 0.5, 0.7, 0.9],
                min_crop_sample_coverages=[0.0, 0.0, 0.0, 0.0, 0.0],
                min_crop_scales=[0.3, 0.3, 0.3, 0.3, 0.3],
                num_crop_sampler=5,
                inter_method=10,
                rand_mirror_prob=0.5,
                shuffle=True
            )
        else:
            self.data=mx.io.ImageDetRecordIter(
                path_imgrec=args.val_rec,
                batch_size=args.batch_size,
                data_shape=data_shape,
                mean_r=123.68,
                mean_g=116.779,
                mean_b=103.939,
                label_pad_width=420,
                shuffle=False
            )
        self._read_data()
        self.reset()

    @property
    def provide_data(self):
        return self.data.provide_data

    @property
    def provide_label(self):
        return self.new_provide_label

    def reset(self):
        self.data.reset()

    def _read_data(self):
        self._data_batch = next(self.data)
        if self._data_batch is None:
            return False
        else:
            original_label = self._data_batch.label[0]
            original_label_length = original_label.shape[1]
            label_head_length = int(original_label[0][4].asscalar())
            object_label_length = int(original_label[0][5].asscalar())
            label_start_idx = 4+label_head_length
            label_num = (original_label_length-
                         label_start_idx+1)//object_label_length
            self.new_label_shape = (self.args.batch_size, label_num,
                                    object_label_length)
            self.new_provide_label = [(self.args.label_name,
                                       self.new_label_shape)]
            new_label = original_label[:,label_start_idx:
                                object_label_length*label_num+label_start_idx]
            self._data_batch.label = [new_label.reshape((-1,label_num,
                                                         object_label_length))]
        return True

    def iter_next(self):
        return self._read_data()

    def next(self):
        if self.iter_next():
            return self._data_batch
        else:
            raise StopIteration