import gluoncv
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from PIL import Image
import numpy as np
import mxnet as mx
import os
import pickle
import time


net = model_zoo.get_model('faster_rcnn_fpn_resnet50_v1b_coco', pretrained=True)

class_list = net.classes

dir_list = ["love"]

for i in range(9):
    start = time.time()
    step = 0
    os.chdir('/hdd/user16')
    file_name = dir_list[i] + '.txt'
    dir_path = '/hdd/user16/img(0~9)/' + dir_list[i]
    os.chdir(dir_path)
    file_names = os.listdir()
    print(file_names)
    for file_name_ in file_names:
        f.write(file_name_+' - ')
        step += 1
        img_path = '/hdd/user16/img(0~9)/' + dir_list[i] + '/' + str(file_name_)
        x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(img_path)
        box_ids, scores, bboxes = net(x)

        for id_num, id in enumerate(box_ids[0]):
            if id != -1:
                if scores[0][id_num] > 0.5:
                    f.write(class_list[int(id.asnumpy())] + '\t')
        f.write('\n')
        if step % 10 == 0:
            print("time for ten : ", time.time() - start)
