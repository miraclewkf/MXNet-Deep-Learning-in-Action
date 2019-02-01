import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_anchors(anchors, sizeNum, ratioNum):
    img = mx.img.imread("anchor_demo/000001.jpg")
    height, width, _ = img.shape
    fig, ax = plt.subplots(1)
    ax.imshow(img.asnumpy())
    edgecolors = ['r','g','y','b']
    for h_i in range(anchors.shape[0]):
        for w_i in range(anchors.shape[1]):
            for index, anchor in enumerate(anchors[h_i,w_i,:,:].asnumpy()):
                xmin = anchor[0]*width
                ymin = anchor[1]*height
                xmax = anchor[2]*width
                ymax = anchor[3]*height
                rect = patches.Rectangle(xy=(xmin,ymin), width=xmax-xmin,
                                         height=ymax-ymin,
                                         edgecolor=edgecolors[index], 
                                         facecolor='None',
                                         linewidth=1.5)
                ax.add_patch(rect)
    plt.savefig("anchor_demo/mapSize_{}*{}_sizeNum_{}_ratioNum_{}.png".format(
                anchors.shape[0], anchors.shape[1], sizeNum, ratioNum))

######################### 2*2, single size and ratio ###############################
input_h = 2
input_w = 2
input = mx.nd.random.uniform(shape=(1,3,input_h,input_w))
anchors = mx.nd.contrib.MultiBoxPrior(data=input, sizes=[0.3], ratios=[1])
print(anchors)
anchors = anchors.reshape((input_h,input_w,-1,4))
print(anchors.shape)
plot_anchors(anchors=anchors, sizeNum=1, ratioNum=1)

######################### 2*2, single size and more ratios ###############################
input_h = 2
input_w = 2
input = mx.nd.random.uniform(shape=(1,3,input_h,input_w))
anchors = mx.nd.contrib.MultiBoxPrior(data=input, sizes=[0.3], ratios=[1,2,0.5])
anchors = anchors.reshape((input_h,input_w,-1,4))
print(anchors.shape)
plot_anchors(anchors=anchors, sizeNum=1, ratioNum=3)

######################### 2*2, more sizes and more ratios ###############################
input_h = 2
input_w = 2
input = mx.nd.random.uniform(shape=(1,3,input_h,input_w))
anchors = mx.nd.contrib.MultiBoxPrior(data=input, sizes=[0.3,0.4],
                                      ratios=[1,2,0.5])
anchors = anchors.reshape((input_h,input_w,-1,4))
print(anchors.shape)
plot_anchors(anchors=anchors, sizeNum=2, ratioNum=3)

######################### 5*5 ###############################
input_h = 5
input_w = 5
input = mx.nd.random.uniform(shape=(1,3,input_h,input_w))
anchors = mx.nd.contrib.MultiBoxPrior(data=input, sizes=[0.1,0.15],
                                      ratios=[1,2,0.5])
anchors = anchors.reshape((input_h,input_w,-1,4))
print(anchors.shape)
plot_anchors(anchors=anchors, sizeNum=2, ratioNum=3)
