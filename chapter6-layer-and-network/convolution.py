import mxnet as mx

input_data = mx.nd.arange(1,51).reshape((1,2,5,5))
print("Input data:")
print(input_data )
weight = mx.nd.arange(1,37).reshape((2,2,3,3))
print("Convolution weight:")
print(weight)
bias = mx.nd.ones(2)
print("Convolution bias:")
print(bias)
output_data = mx.nd.Convolution(data=input_data, weight=weight, bias=bias, 
	                            kernel=(3,3), stride=(1,1), pad=(0,0), 
	                            dilate=(1,1), num_filter=2, num_group=1)
print("Regular convolution result:")
print(output_data)
output_data_dilate = mx.nd.Convolution(data=input_data, weight=weight, bias=bias,
                                       kernel=(3,3), stride=(1,1), pad=(0,0),
                                       dilate=(2,2), num_filter=2, num_group=1)
print("Dilated convolution result:")
print(output_data_dilate)
weight = mx.nd.arange(1,19).reshape((2,1,3,3))
print("Group convolution weight:")
print(weight)
output_data_group = mx.nd.Convolution(data=input_data, weight=weight, bias=bias,
                                      kernel=(3,3), stride=(1,1), pad=(0,0),
                                      dilate=(1,1), num_filter=2, num_group=2)
print("Group convolution result:")
print(output_data_group)

