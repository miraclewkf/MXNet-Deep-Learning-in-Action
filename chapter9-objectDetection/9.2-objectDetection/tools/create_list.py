import os
import argparse
from PIL import Image
import xml.etree.ElementTree as ET
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--save-path', type=str, default='')
    parser.add_argument('--dataset-path', type=str, default='')
    parser.add_argument('--shuffle', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    label_dic = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5,
                 "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11,
                 "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16,
                 "sofa": 17, "train": 18, "tvmonitor": 19}
    args = parse_args()
    if not os.path.exists(os.path.join(args.save_path, "{}.lst".format(args.set))):
        os.mknod(os.path.join(args.save_path, "{}.lst".format(args.set)))
    with open(os.path.join(args.save_path, "{}.txt".format(args.set)), "r") as input_file:
        lines = input_file.readlines()
        if args.shuffle:
            random.shuffle(lines)
        with open(os.path.join(args.save_path, "{}.lst".format(args.set)), "w") as output_file:
            index = 0
            for line in lines:
                line = line.strip()
                out_str = "\t".join([str(index), "2", "6"])
                img = Image.open(os.path.join(args.dataset_path, "JPEGImages", line+".jpg"))
                width, height = img.size
                xml_path = os.path.join(args.dataset_path, "Annotations", line+".xml")
                tree = ET.parse(xml_path)
                root = tree.getroot()
                objects = root.findall('object')
                for object in objects:
                    name = object.find('name').text
                    difficult = ("%.4f" % int(object.find('difficult').text))
                    label_idx = ("%.4f" % label_dic[name])
                    bndbox = object.find('bndbox')
                    xmin = ("%.4f" % (int(bndbox.find('xmin').text)/width))
                    ymin = ("%.4f" % (int(bndbox.find('ymin').text)/height))
                    xmax = ("%.4f" % (int(bndbox.find('xmax').text)/width))
                    ymax = ("%.4f" % (int(bndbox.find('ymax').text)/height))
                    object_str = "\t".join([label_idx, xmin, ymin, xmax, ymax, difficult])
                    out_str = "\t".join([out_str, object_str])
                out_str = "\t".join([out_str, "{}/JPEGImages/".format(args.dataset_path.split("/")[-1])+line+".jpg"+"\n"])
                output_file.writelines(out_str)
                index += 1

if __name__ == '__main__':
    main()
