import gluoncv
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from PIL import Image
import numpy as np
import mxnet as mx
import os
import pickle
import time

Path = r'C:\Users\Lee Wook Jin\PycharmProjects\Fast-RCNN'

net = model_zoo.get_model('faster_rcnn_fpn_resnet50_v1b_coco', pretrained=True)

class_list = net.classes

img_folders = ['not_completed']

for folder in img_folders:
    Path_added = Path + '/' + folder
    os.chdir(Path_added)
    img_sub_folders = os.listdir()
    print(img_sub_folders)
    for sub_folder in img_sub_folders:
        print(sub_folder)
        start = time.time()
        Path_added_2 = Path_added + '/' + sub_folder
        step = 0
        os.chdir(Path_added_2)
        img_files = os.listdir()
        file_save_path = Path + '/' + sub_folder + '.txt'
        f = open(file_save_path, 'w')
        for img in img_files:
            img_path = Path_added_2 + '/' + img
            f.write(img + ' - ')
            step += 1

            x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(img_path)
            box_ids, scores, bboxes = net(x)

            for id_num, id in enumerate(box_ids[0]):
                if id != -1:
                    if scores[0][id_num] > 0.5:
                        f.write(class_list[int(id.asnumpy())] + '\t')
            f.write('\n')
            if step % 10 == 0:
                print("time for ten : ", time.time() - start)


        f.close()
