import cv2
import os
import shutil
import numpy as np

def split_sets(opt):

    split_dict = {"train": opt.train,
                  "val": opt.val,
                  "test": opt.test}
    src_dir = opt.src
    for key in split_dict:
        file1 = open(split_dict[key], 'r')
        Lines = file1.readlines()
        file1.close()
        for line in Lines:
            y, x = line.split('/')[0], line.split('/')[1].replace("\n","")
            src = src_dir + str(x) + ".webm"
            dir_name = "dataset/" + key + "/" + y
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            shutil.copy2(src, dir_name)


def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

def get_center_crop(image, h, w):
    center = image.shape / 2
    x = center[1] - w / 2
    y = center[0] - h / 2

    crop_img = image[int(y):int(y + h), int(x):int(x + w)]
    return crop_img

def return_frames(dir, split_type):
    cap = cv2.VideoCapture(dir)
    a = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret == True):
            frame = cv2.resize(frame, (256, 256))
            #cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            if (split_type == "test"):
                frame = get_center_crop(frame, 224, 224)
            else:
                frame = get_random_crop(frame, 224, 224)
            frame = np.transpose(frame, (2, 0, 1))
            a.append(frame/255)
        else:
            break
    return a