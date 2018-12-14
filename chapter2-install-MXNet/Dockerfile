FROM mxnet/python:1.2.1_gpu_cuda90

RUN apt-get update && apt-get install -y python3 python3-pip \
                   && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade mxnet-cu90==1.2.1

