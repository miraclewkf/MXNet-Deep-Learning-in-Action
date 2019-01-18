import mxnet as mx
input_data = mx.nd.cast(mx.nd.arange(0.1,2.1,0.1).reshape((5,4)), 'float16')
print("Input data:")
print(input_data)
out_data = mx.nd.softmax(data=input_data, axis=-1)
print("Softmax result:")
print(out_data)

label = mx.nd.array([0,0,0,0,0], dtype='float16')
ce_loss = mx.nd.softmax_cross_entropy(data=input_data, label=label)
print("All predictions are wrong, cross entropy loss:")
print(ce_loss)

label = mx.nd.array([1,0,3,3,0], dtype='float16')
ce_loss = mx.nd.softmax_cross_entropy(data=input_data, label=label)
print("Part predictions are wrong, cross entropy loss:")
print(ce_loss)

label = mx.nd.array([3,3,3,3,3], dtype='float16')
ce_loss = mx.nd.softmax_cross_entropy(data=input_data, label=label)
print("All predictions are right, cross entropy loss:")
print(ce_loss)

smoothl1_loss = mx.nd.smooth_l1(data=input_data, scalar=1)
print("SmoothL1 loss:")
print(smoothl1_loss)