import mxnet as mx
input_data = mx.nd.arange(1,19).reshape((1,2,3,3))
print("Input data:")
print(input_data)
weight = mx.nd.arange(1,73).reshape((4,18))
print("FullyConnected weight:")
print(weight)
bias = mx.nd.ones(4)
print("FullyConnected bias:")
print(bias)
out_data = mx.nd.FullyConnected(data=input_data, weight=weight, bias=bias, 
	                        num_hidden=4, flatten=1)
print("FullyConnected result:")
print(out_data)
