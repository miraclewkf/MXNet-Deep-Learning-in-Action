from gluoncv import data, utils
import matplotlib.pyplot as plt

train_data = data.ImageNet(root='data/ILSVRC2012',
                           train=True)
val_data = data.ImageNet(root='data/ILSVRC2012',
                         train=False)

# Number of data
print("Number of train data: {}".format(len(train_data)))
print("Number of validation data: {}".format(len(val_data)))

# One image
image, ground_truth = train_data[0]
print("Shape of image: {}".format(image.shape))

# Visualization
utils.viz.plot_image(img=image)
plt.savefig('imagenet.png')
