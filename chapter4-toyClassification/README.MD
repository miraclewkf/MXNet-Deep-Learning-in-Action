### 第4章 MNIST手写数字体分类

这一章以MNIST手写数字体数据集为例介绍如何通过MXNet框架实现图像分类。

#### 项目组成介绍

* 代码清单4-1对应`train_mnist_code4-1.py`脚本。
* 代码清单4-2对应`test_mnist_code4-2.py`脚本。
* 训练得到的模型保存在`output`文件夹下。
* 测试模型时用的测试图像保存在`test_image`文件夹下。
* `train-images.idx3-ubyte`和`train-labels.idx1-ubyte`是训练数据及标签文件，`t10k-images.idx3-ubyte`和`t10k-labels.idx1-ubyte`是验证数据及标签文件。
