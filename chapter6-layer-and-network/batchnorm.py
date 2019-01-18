import mxnet as mx
input_data = mx.nd.arange(1,9).reshape((1,2,2,2))
print("Input data:")
print(input_data)
gamma = mx.nd.ones(2)
print("gamma:")
print(gamma)
beta = mx.nd.ones(2)
print("beta:")
print(beta)
moving_mean = mx.nd.ones(2)*3
print("moving_mean:")
print(moving_mean)
moving_var = mx.nd.ones(2)*2
print("moving_var:")
print(moving_var)
out_data = mx.nd.BatchNorm(data=input_data, gamma=gamma, beta=beta,
                           moving_mean=moving_mean, moving_var=moving_var,
                           momentum=0.9, fix_gamma=1, use_global_stats=1)
print("BatchNorm result:")
print(out_data)
