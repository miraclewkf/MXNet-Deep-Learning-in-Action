import mxnet as mx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def parse_xml(xml_path):
    bbox= []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    for object in objects:
        name = object.find('name').text
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox_i = [name, xmin, ymin, xmax, ymax]
        bbox.append(bbox_i)
    return bbox

def visiualize_bbox(image, bbox, name):
    fig, ax = plt.subplots()
    plt.imshow(image)
    colors = dict()
    for bbox_i in bbox:
        cls_name = bbox_i[0]
        if cls_name not in colors:
            colors[cls_name] = (random.random(), random.random(),
                                random.random())
        xmin = bbox_i[1]
        ymin = bbox_i[2]
        xmax = bbox_i[3]
        ymax = bbox_i[4]
        rect = patches.Rectangle(xy=(xmin,ymin), width=xmax-xmin,
                                 height=ymax-ymin, 
                                 edgecolor=colors[cls_name],
                                 facecolor='None',
                                 linewidth=3.5)
        plt.text(xmin, ymin-2, '{:s}'.format(cls_name),
                 bbox=dict(facecolor=colors[cls_name], 
                 alpha=0.5))
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig('VOC_image_demo/{}_gt.png'.format(name))
    plt.close()

if __name__ == '__main__':
    name = '000001'
    xml_path = 'VOC_image_demo/{}.xml'.format(name)
    img_path = 'VOC_image_demo/{}.jpg'.format(name)
    bbox = parse_xml(xml_path=xml_path)
    image_string = open(img_path, 'rb').read()
    image = mx.image.imdecode(image_string, flag=1).asnumpy()
    visiualize_bbox(image, bbox, name)