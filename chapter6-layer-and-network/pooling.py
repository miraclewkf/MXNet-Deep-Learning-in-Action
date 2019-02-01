import mxnet as mx
input_data = mx.nd.arange(1,51).reshape((1,2,5,5))
print("Input data:")
print(input_data)
out_data = mx.nd.Pooling(data=input_data, kernel=(2,2), pool_type='max',
                         global_pool=0, pooling_convention='valid', 
                         stride=(1,1), pad=(0,0))
print("Max pooling result:")
print(out_data)
out_data = mx.nd.Pooling(data=input_data, kernel=(2,2), pool_type='avg', 
	                 global_pool=0, pooling_convention='valid',
	                 stride=(1,1), pad=(0,0))
print("Avg pooling result:")
print(out_data)
out_data = mx.nd.Pooling(data=input_data, kernel=(2,2), pool_type='max', 
	                 global_pool=1, pooling_convention='valid', 
	                 stride=(1,1), pad=(0,0))
print("Global max pooling result:")
print(out_data)

