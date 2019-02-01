import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_anchors(anchors, img, text, linestyle='-'):
    height, width, _ = img.shape
    colors = ['r','y','b','c','m']
    for num_i in range(anchors.shape[0]):
        for index, anchor in enumerate(anchors[num_i,:,:].asnumpy()):
            xmin = anchor[0]*width
            ymin = anchor[1]*height
            xmax = anchor[2]*width
            ymax = anchor[3]*height
            rect = patches.Rectangle(xy=(xmin,ymin), width=xmax-xmin,
                                     height=ymax-ymin, edgecolor=colors[index],
                                     facecolor='None', linestyle=linestyle,
                                     linewidth=1.5)
            ax.text(xmin, ymin, text[index],
                    bbox=dict(facecolor=colors[index], alpha=0.5))
            ax.add_patch(rect)

img = mx.img.imread("target_demo/000001.jpg")
fig,ax = plt.subplots(1)
ax.imshow(img.asnumpy())

ground_truth = mx.nd.array([[[0, 0.136,0.48,0.552,0.742],
                             [1, 0.023,0.024,0.997,0.996]]])
plot_anchors(anchors=ground_truth[:, :, 1:], img=img,
             text=['dog','person'])

anchor = mx.nd.array([[[0.1, 0.3, 0.4, 0.6],
                       [0.15, 0.1, 0.85, 0.8],
                       [0.1, 0.2, 0.6, 0.4],
                       [0.25, 0.5, 0.55, 0.7],
                       [0.05, 0.08, 0.95, 0.9]]])
plot_anchors(anchors=anchor, img=img, text=['1','2','3','4','5'],
             linestyle=':')

plt.savefig("target_demo/anchor_gt.png")

cls_pred = mx.nd.array([[[0.4, 0.3, 0.2, 0.1, 0.1],
                         [0.6, 0.7, 0.8, 0.9, 0.9]]])
tmp = mx.nd.contrib.MultiBoxTarget(anchor=anchor,
                                   label=ground_truth,
                                   cls_pred=cls_pred,
                                   overlap_threshold=0.5,
                                   ignore_label=-1,
                                   negative_mining_ratio=3,
                                   variances=[0.1,0.1,0.2,0.2])
print("location target: {}".format(tmp[0]))
print("location target mask: {}".format(tmp[1]))
print("classification target: {}".format(tmp[2]))