### 第5章 数据读取及增强

这一章介绍MXNet的数据读取和增强操作，数据读取部分包含基于原图像读取和基于RecordIO文件读取两种方式，数据增强操作包括resize、crop、镜像、色彩变换等。

#### 项目组成介绍

* 数据（公鸡和鸵鸟）下载链接：[GoogleDrive](https://drive.google.com/open?id=1hhf4slsGMAyh94kTLyDz6GS2TyloTROB) or [BaiDuYun](https://pan.baidu.com/s/1Ze2307WG9l-rbK9yb_cExQ)(提取码: xati)
* 数据处理过程中用到的辅助脚本保存在`tools`文件夹中，主要是`im2rec.py`脚本。
* 5.1节基于原图像读取数据的代码保存在`read_lst.py`脚本中。
* 5.2节基于RecordIO文件读取数据的代码保存在 `read_rec.py`脚本中。
* 5.3节关于数据增强的代码和图像保存在`5.3-data-augmentation`文件夹中。
