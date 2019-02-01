import os

for file in ["train", "val"]:
    index = 1
    if not os.path.exists("VOC2012/{}.lst".format(file)):
        os.mknod("VOC2012/{}.lst".format(file))
    with open("VOC2012/{}.lst".format(file), "w") as output_file:
        with open("VOC2012/ImageSets/Segmentation/{}.txt".format(file), "r") as input_file:
            lines = input_file.readlines()
            for line in lines:
                img_name = line.strip()
                out_line = str(index) + "\t" + "JPEGImages/{}.jpg".format(img_name) + "\t" + "SegmentationClass/{}.png".format(img_name) + "\n"
                output_file.writelines(out_line)
                index += 1