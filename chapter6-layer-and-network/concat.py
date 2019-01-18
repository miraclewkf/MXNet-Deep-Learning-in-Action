import mxnet as mx
input_data1 = mx.nd.arange(1,51).reshape((1,2,5,5))
print("Input data1:")
print(input_data1)
input_data2 = mx.nd.arange(5,31).reshape((1,1,5,5))
print("Input data2:")
print(input_data2)
out_data = mx.nd.concat(input_data1,input_data2,dim=1)
print("Concat result:")
print(out_data)