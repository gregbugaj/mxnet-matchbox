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
fh = logging.FileHandler('image-classification.log')
logger = logging.getLogger()
logger.addHandler(fh)
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug('\n%s', '-' * 100)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)

# Dependency update
# Build dependencies ar only requred on new system
# python3 -m pip install --upgrade pip
# python3 -m pip install scikit-build
# python3 -m pip install cmake
# python3 -m pip install -r requirements.txt

# Once the images are loaded, we need to ensure the images are of the same size.
# We will resize all the images to be 224 * 224 pixels.
# Let’s flip 50% of the training data set horizontally and crop them to 224 * 224 pixels
train_augs = [
    mx.image.HorizontalFlipAug(.5),
    mx.image.RandomCropAug((224, 224))
]

# For the validation and test data sets, let’s center crop to get each image to 224 224.
# All the images in the train, test, and validation data sets will now be of 224 224 size.
val_test_augs = [
    mx.image.CenterCropAug((224, 224))
]


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


def separateTrainingSet():
    print("Seperating traing set")

    test_data = os.path.join("./trainingset", "test_data")
    train_data = os.path.join("./trainingset", "train_data")
    val_data = os.path.join("./trainingset", "val_data")

    if not os.path.exists(test_data):
        os.makedirs(test_data)
    if not os.path.exists(train_data):
        os.makedirs(train_data)
    if not os.path.exists(val_data):
        os.makedirs(val_data)

    root = "./dataset"
    classes = os.listdir(root)
    classes.sort()
    print("Classes : {}".format(classes))

    for clazz in classes:

        print(clazz)
        clazzDir = os.path.join(root, clazz)
        filenames = os.listdir(clazzDir)
        #random.shuffle(filenames)
        filenames.sort()

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

        trainSetData = os.path.join("./trainingset/train_data", clazz)
        testSetData = os.path.join("./trainingset/test_data", clazz)
        valSetData = os.path.join("./trainingset/val_data", clazz)

        copyfiles(training_files, clazzDir, trainSetData)
        copyfiles(validation_files, clazzDir, valSetData)
        copyfiles(testing_files, clazzDir, testSetData)


def get_imagenet_transforms(data_shape=224, dtype='float32'):
    def train_transform(data, label):
        data = data.astype('float32')
        augs = [
            mx.image.HorizontalFlipAug(.5),
            mx.image.RandomCropAug((224, 224))
        ]
        for aug in augs:
            data = aug(data)
        # from (H x W x c) to (c x H x W)
        data = mx.nd.transpose(data, (2, 0, 1))
 
        # Normalzie 0..1 range
        data = data.astype('float32') / 255

        return data, mx.nd.array(([label])).asscalar().astype('float32')

    def val_transform(data, label):
        data = data.astype('float32')
        augs = [
            mx.image.CenterCropAug((224, 224))
        ]
        for aug in augs:
            data = aug(data)
        # from (H x W x c) to (c x H x W)
        data = mx.nd.transpose(data, (2, 0, 1))

        # Normalzie 0..1 range
        data = data.astype('float32') / 255

        return data, mx.nd.array(([label])).asscalar().astype('float32')

    def __train_transform(image, label):
        image, _ = mx.image.random_size_crop(image, (data_shape, data_shape), 0.08, (3/4., 4/3.))
        image = mx.nd.image.random_flip_left_right(image)
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            
        return mx.nd.cast(image, dtype), label

    def __val_transform(image, label):
        image = mx.image.resize_short(image, data_shape + 32)
        image, _ = mx.image.center_crop(image, (data_shape, data_shape))
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(
            0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return mx.nd.cast(image, dtype), label

    return train_transform, val_transform, val_transform


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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Diagnose script for checking the current system.')
    parser.add_argument('--region', default='', type=str, help="Additional sites in which region(s) to test. Specify 'cn' for example to test mirror sites in China.")
    parser.add_argument('--timeout', default=10, type=int, help="Connection test timeout threshold, 0 to disable.")
    
    args = parser.parse_args()
    return args

def get_build_features_str():
    import mxnet.runtime
    features = mxnet.runtime.Features()
    return '\n'.join(map(str, list(features.values())))

def check_mxnet():
    print('----------MXNet Info-----------')
    try:
        import mxnet
        print('Version      :', mxnet.__version__)
        mx_dir = os.path.dirname(mxnet.__file__)
        print('Directory    :', mx_dir)
        commit_hash = os.path.join(mx_dir, 'COMMIT_HASH')
        if os.path.exists(commit_hash):
            with open(commit_hash, 'r') as f:
                ch = f.read().strip()
                print('Commit Hash   :', ch)
        else:
            print('Commit hash file "{}" not found. Not installed from pre-built package or built from source.'.format(commit_hash))
        print('Library      :', mxnet.libinfo.find_lib_path())
        try:
            print('Build features:')
            print(get_build_features_str())
        except Exception:
            print('No runtime build feature info available')
    except ImportError:
        print('No MXNet installed.')
    except Exception as e:
        import traceback
        if not isinstance(e, IOError):
            print("An error occured trying to import mxnet.")
            print("This is very likely due to missing missing or incompatible library files.")
        print(traceback.format_exc())

if __name__ == "__main__":
    args = parse_args()

    #extractLogosToFolders()
    #separateTrainingSet()
