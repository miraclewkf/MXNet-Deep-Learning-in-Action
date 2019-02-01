import mxnet as mx
from mxnet.gluon.data import vision

input_data = mx.nd.random.uniform(0,255,shape=(2,4,3)).astype('uint8')
print(input_data)
transformer_tensor = vision.transforms.ToTensor()
tensor_data = transformer_tensor(input_data)
print(tensor_data)

transformer_normalize = vision.transforms.Normalize(mean=0.13, std=0.31)
normal_data = transformer_normalize(tensor_data)
print(normal_data)