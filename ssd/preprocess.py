import argparse
import os
import cv2
import pandas as pd

from mxnet import autograd as ag
import mxnet as mx
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.contrib.io import DataLoaderIter
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageFolderDataset
from gluoncv.model_zoo import get_model

import time
import random
import os
import argparse
import cv2

import numpy as np
import shutil
import math
import random
from os import listdir
from os.path import isfile, join
from os import walk
from time import time
import matplotlib.pyplot as plt

import logging
import tarfile
logging.basicConfig(level=logging.INFO)

# logging
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('ssd-preprocess.log')
logger = logging.getLogger()
logger.addHandler(fh)
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug('\n%s', '-' * 100)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)


## https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv


def extractLogosToFolders():
    print("Image training set segregation")
    annotations = open(
        'flickr_logos_27_dataset_training_set_annotation.txt', 'r')
    count = 0

    while True:
        count += 1
        line = annotations.readline()
        if not line:
            break

        line = line.strip()
        parts = line.split(" ")
        print("Line {} : {} == {}".format(count, line, parts))
        
        # 144503924.jpg  Adidas 1 38 12 234 142
        # 2662264721.jpg RedBull 2 3 197 3 197
        name = parts[0]
        label = parts[1].lower()
        clazz = int(parts[2])
        x1 = int(parts[3])
        y1 = int(parts[4])
        x2 = int(parts[5])
        y2 = int(parts[6])
        w = x2 - x1
        h = y2 - y1

        print(parts)
        print(" {} : {} ==  {} : {} ".format(x1, y1, x2, y2))
        print(" {} : {} ".format(w, h))

        # 2662264721.jpg RedBull 2 3 197 3 197
        if w == 0 or h == 0:
            continue

        imagePath = os.path.join("./flickr_logos_27_dataset_images", name)
        labelDir = os.path.join("./dataset", label)
        imageDestName = os.path.join("./dataset", label, name)
        if not os.path.exists(labelDir):
            os.makedirs(labelDir)

        img = cv2.imread(imagePath)
        crop = img[y1:y1+h, x1:x1+w]

        #cv2.imshow('Image', crop)
        #cv2.waitKey(100)
        print(imageDestName)
        # print (img)
        cv2.imwrite(imageDestName, crop)


def copyfiles(files, srcDir, destDir):
    if not os.path.exists(destDir):
        os.makedirs(destDir)

    for filename in files:
        src = os.path.join(srcDir, filename)
        dest = os.path.join(destDir, filename)
        print("copy   > {} : {}".format(src, dest))
        shutil.copy(src, dest)


def separateTrainingSet(data_in_dir, data_out_dir):
    print("Seperating traing set")
    root = data_in_dir

    test_data = os.path.join(data_out_dir, "test_data")
    train_data = os.path.join(data_out_dir, "train_data")
    val_data = os.path.join(data_out_dir, "val_data")

    if not os.path.exists(test_data):
        os.makedirs(test_data)
    if not os.path.exists(train_data):
        os.makedirs(train_data)
    if not os.path.exists(val_data):
        os.makedirs(val_data)

    
    filenames = os.listdir(data_in_dir)
    random.shuffle(filenames)
    # filenames.sort()

    size = len(filenames)
    validationSize = math.ceil(size * 0.3)  # 30 percent validation size
    testSize = math.ceil(size * 0.1)  # 10 percent testing size
    trainingSize = size - validationSize - testSize  # 60 percent training
    print("Class >>  size/training/validation/test : {}, {}, {}, {}".format(size,
                                                                            trainingSize, validationSize, testSize))
    validation_files = filenames[:validationSize]
    testing_files = filenames[validationSize:validationSize+testSize]
    training_files = filenames[validationSize+testSize:]

    print("Number of Validation Images : {}, {}".format(
        len(validation_files), validation_files))
    print("Number of Testing Images    : {}, {}".format(
        len(testing_files), testing_files))
    print("Number of Training Images   : {}, {}".format(
        len(training_files), training_files))

    trainSetData = os.path.join(data_out_dir, "train_data")
    testSetData = os.path.join(data_out_dir, "test_data")
    valSetData = os.path.join(data_out_dir, "val_data")

    copyfiles(training_files, data_in_dir, trainSetData)
    copyfiles(validation_files, data_in_dir, valSetData)
    copyfiles(testing_files, data_in_dir, testSetData)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def resizeRawImages(data_root_dir):
    print("Resizing traing set")   
    in_data = os.path.join(data_root_dir, "raw")
    out_data = os.path.join(data_root_dir, "converted")

    if not os.path.exists(out_data):
        os.makedirs(out_data)

    filenames = os.listdir(in_data)
    for  i, name in enumerate(filenames):
        filename = os.path.join(in_data, name)
        filename_out_1 = os.path.join(out_data, "{}_gray.png".format(i))
        filename_out_2 = os.path.join(out_data, "{}_binary.png".format(i))

        img = cv2.imread(filename, 0)
        # img = np.array(img) 
        img = image_resize(img, width=512)
        # Otsu's thresholding after Gaussian filtering
        # blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(filename_out_1, im_rgb)
        cv2.imwrite(filename_out_2, th3)

#    copyfiles(training_files, clazzDir, trainSetData)

import argparse
import os
import cv2
import pandas as pd

def transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = img # cv2.merge([img,img,img])
    # perform transformations on image
    # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)  # EuclideanDistanceTransform
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)  # LinearDistanceTransform
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)   # MaxDistanceTransform
    # merge the transformed channels back to an image
    transformed_image = cv2.merge((b, g, r))
    # cv2.imshow('Image', transformed_image)
    # cv2.waitKey(-1)
    # transformed_image = (255-transformed_image) # Invert
    return transformed_image
    
def process_A(dir_src, dir_dest):
    dirs = os.listdir(dir_src)
    dirs.sort()
    print("Classes : {}".format(dirs))

    for clazz in dirs:
        print(clazz)
        clazz_dir = os.path.join(dir_src, clazz)
        clazz_dir_dest = os.path.join(dir_dest, clazz)

        filenames = os.listdir(clazz_dir)
        #random.shuffle(filenames)
        filenames.sort()
        size = len(filenames)

        print(size)
        print(filenames)
        if not os.path.exists(clazz_dir_dest):
         os.makedirs(clazz_dir_dest)

        for filename in filenames:
            # open image file
            path = os.path.join(clazz_dir, filename)
            path_dest = os.path.join(clazz_dir_dest, filename) + ".png"
            # img = cv2.imread(os.path.join(clazz_dir, filename), cv2.IMREAD_GRAYSCALE)            
            img = cv2.imread(os.path.join(clazz_dir, filename))            
            # img = transform(img)
            img = image_resize(img, height=512)

            ## Merge two images
            # l_img = np.zeros((512, 512), np.uint8)
            l_img = np.ones((512, 512, 3),np.uint8)*255
            s_img = img # np.zeros((512, 512, 3), np.uint8)
            
            x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
            y_offset = 0
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

            img = l_img
            enabled = False
            if enabled:
                kernel = np.ones((2, 2), np.uint8) 
                # Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(img,(5, 5),0)
                ret3,th3 = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img_dilation = cv2.dilate(th3, kernel, iterations=1) 
                im_rgb = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2RGB)
                img = transform(im_rgb)
            # img = transform(blur)
            cv2.imwrite(path_dest, img)

def processB(dir_src, dir_dest):
    dirs = os.listdir(dir_src)
    dirs.sort()
    print("Classes : {}".format(dirs))

    for clazz in dirs:
        print(clazz)
        clazz_dir = os.path.join(dir_src, clazz)
        clazz_dir_dest = os.path.join(dir_dest, clazz)

        filenames = os.listdir(clazz_dir)
        #random.shuffle(filenames)
        filenames.sort()
        size = len(filenames)

        print(size)
        print(filenames)
        if not os.path.exists(clazz_dir_dest):
         os.makedirs(clazz_dir_dest)

        for filename in filenames:
            # open image file
            path = os.path.join(clazz_dir, filename)
            path_dest = os.path.join(clazz_dir_dest, filename) + ".png"
            # img = cv2.imread(os.path.join(clazz_dir, filename), cv2.IMREAD_GRAYSCALE)            
            img = cv2.imread(os.path.join(clazz_dir, filename))            
            img = transform(img)
            img = image_resize(img, height=512)

            ## Merge two images
            # l_img = np.zeros((512, 512), np.uint8)
            l_img = np.ones((512, 512, 3),np.uint8)*255
            s_img = img # np.zeros((512, 512, 3), np.uint8)
            
            x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
            y_offset = 0
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

            img = l_img
            enabled = True
            if enabled:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((2, 2), np.uint8) 
                # Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(img,(5, 5), 0)
                ret3,th3 = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img_dilation = cv2.dilate(th3, kernel, iterations=1) 
                im_rgb = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2RGB)
                # img = transform(im_rgb)

            cv2.imwrite(path_dest, img)

def process(dir_src, dir_dest):
    dirs = os.listdir(dir_src)
    dirs.sort()
    print("Classes : {}".format(dirs))

    for clazz in dirs:
        print(clazz)
        clazz_dir = os.path.join(dir_src, clazz)
        clazz_dir_dest = os.path.join(dir_dest, clazz)

        filenames = os.listdir(clazz_dir)
        #random.shuffle(filenames)
        filenames.sort()
        size = len(filenames)

        print(size)
        print(filenames)
        if not os.path.exists(clazz_dir_dest):
         os.makedirs(clazz_dir_dest)

        for filename in filenames:
            try:
                print (filename)
                # open image file
                path = os.path.join(clazz_dir, filename)
                path_dest = os.path.join(clazz_dir_dest, filename) + ".png"
                # img = cv2.imread(os.path.join(clazz_dir, filename), cv2.IMREAD_GRAYSCALE)            
                img = cv2.imread(os.path.join(clazz_dir, filename))     
                img = transform(img)
                img = image_resize(img, height=512)

                ## Merge two images
                # l_img = np.zeros((512, 512), np.uint8)
                l_img = np.ones((512, 512, 3),np.uint8)*255
                s_img = img # np.zeros((512, 512, 3), np.uint8)
                
                x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
                y_offset = 0
                l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

                img = l_img
    
                cv2.imwrite(path_dest, img)
            except:
                print('')


def imageInfo(srcImage):
    print("image : {}".format(srcImage))
    img = cv2.imread(srcImage)
    cv2.imshow('Image', img)
    cv2.waitKey(10000)


def show_images(imgs, nrows, ncols, figsize=None):
    """plot a grid of images"""
    figsize = (ncols, nrows)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            figs[i][j].imshow(imgs[i*ncols+j].asnumpy())
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


def display(image_data):
    """display images"""
    # Now, display the first 32 images in a 8 * 4 grid:
    for X, _ in image_data:
        # from (B x c x H x W) to (Bx H x W x c)
        X = X.transpose((0, 2, 3, 1)).clip(0, 255) / 255
        show_images(X, 5, 8)
        break

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Image preprocessor')

    parser.add_argument('--data-src', dest='data_dir_src', 
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data', 'images'), type=str)

    parser.add_argument('--data-dest', dest='data_dir_dest', 
                        help='data directory to output images to', default=os.path.join(os.getcwd(), 'data', 'out'), type=str)

    args = parser.parse_args()
    return args
                
if __name__ == "__main__":
    args = parse_args()

    print(args.data_dir_src)
    print(args.data_dir_dest)
    
    process(args.data_dir_src, args.data_dir_dest)

    data_root_dir = "./data/hicfa"
    data_out_dir = "./data/hicfa-training"
    # resizeRawImages(data_root_dir)
    # separateTrainingSet(data_in_dir = os.path.join(data_root_dir, "converted"), data_out_dir = data_out_dir)

    #extractLogosToFolders()
    #separateTrainingSet()

