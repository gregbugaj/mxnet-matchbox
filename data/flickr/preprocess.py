import time
import random
import os
import argparse
import mxnet as mx
import cv2

import shutil
import math
import random
from os import listdir
from os.path import isfile, join
from os import walk

import matplotlib.pyplot as plt

# Dependency update
# python3 -m pip install --upgrade pip
# python3 -m pip install  opencv-python
# python3 -m pip install  python3 -m pip install  matplotlib


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
    annotations = open('flickr_logos_27_dataset_training_set_annotation.txt', 'r') 
    count = 0

    while True:
        count += 1
        line = annotations.readline()
        if not line:
            break

        line = line.strip()
        parts = line.split(" ")
        print ("Line {} : {} == {}".format(count, line, parts))

        # 144503924.jpg Adidas 1 38 12 234 142 
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

        print (parts)
        print (" {} : {} ==  {} : {} ".format(x1, y1, x2, y2))
        print (" {} : {} ".format(w, h))

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
        src  = os.path.join(srcDir, filename)
        dest = os.path.join(destDir, filename)
        print ("copy   > {} : {}".format(src, dest))
        
        shutil.copy (src, dest)

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
        validationSize = math.ceil(size * 0.3) # 30 percent validation size
        testSize = math.ceil(size * 0.1) # 10 percent testing size
        trainingSize = size - validationSize - testSize  # 60 percent training
        print("Class >>  size/training/validation/test : {}, {}, {}, {}".format(size,trainingSize, validationSize, testSize))         
        validation_files = filenames[:validationSize]
        testing_files    = filenames[validationSize:validationSize+testSize]
        training_files   = filenames[validationSize+testSize:]

        print("Validation  >> {}, {}".format(len(validation_files), validation_files))
        print("Testing     >> {}, {}".format(len(testing_files), testing_files))
        print("Training    >> {}, {}".format(len(training_files), training_files))
        
        trainSetData = os.path.join("./trainingset/train_data", clazz)
        testSetData = os.path.join("./trainingset/test_data", clazz)
        valSetData = os.path.join("./trainingset/val_data", clazz)

        copyfiles(training_files, clazzDir, trainSetData)
        copyfiles(validation_files, clazzDir, valSetData)
        copyfiles(testing_files, clazzDir, testSetData)

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    # from (H x W x c) to (c x H x W)
    data = mx.nd.transpose(data, (2, 0, 1))
    return data, mx.nd.array(([label])).asscalar().astype('float32')


def imageInfo(srcImage):
    print("image : {}".format(srcImage))
    img = cv2.imread(srcImage)
    cv2.imshow('Image', img)
    cv2.waitKey(10000)
 
def show_images(imgs, nrows, ncols, figsize=None):

    print(imgs)
    """plot a grid of images"""
    figsize = (ncols, nrows)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            figs[i][j].imshow(imgs[i*ncols+j].asnumpy())
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


def train():
    print("Training")    

    # MXNetError: [11:31:23] ../src/io/batchify.cc:128: Check failed: ashape == inputs[j][i].shape() ([394,395,3] vs. [36,49,3])
    # StackBatchify requires all data along batch dim to be the same, mismatch [394,395,3] vs. [36,49,3]

    batch_size = 1
    num_classes = 3
    num_epochs = 1
    num_gpu = 1
    ctx = mx.cpu()#[mx.gpu(i) for i in range(num_gpu)]
    print (ctx)

    # performs the transformation on the data and returns the updated data set
    data_directory = './trainingset/'
    # Transform the image when loading
    train_imgs = mx.gluon.data.vision.ImageFolderDataset(data_directory+'train_data', transform=lambda X, y: transform(X, y, train_augs))
   # test_imgs  = mx.gluon.data.vision.ImageFolderDataset(data_directory+'test_data', transform=lambda X, y: transform(X, y, val_test_augs))
    #val_imgs   = mx.gluon.data.vision.ImageFolderDataset(data_directory+'val_data', transform=lambda X, y: transform(X, y, val_test_augs))
   
    # print(train_imgs)
    # print(test_imgs)
   #  print(val_imgs)

    train_data = mx.gluon.data.DataLoader(train_imgs, batch_size, num_workers=1, shuffle=True)
   # val_data = mx.gluon.data.DataLoader(val_imgs, batch_size, num_workers=1)
    #test_data = mx.gluon.data.DataLoader(test_imgs, batch_size, num_workers=1)
       
    print(train_data)
    # Now, display the first 32 images in a 8 * 4 grid:
    for X, _ in train_data:
        # from (B x c x H x W) to (Bx H x W x c)
        #Y = X.transpose((0, 2, 3, 1)).clip(0, 255)/255
        print (X)
        #show_images(X, 4, 8)
        break

   # print(val_data)
   # print(test_data)

if __name__ == "__main__":
    #extractLogosToFolders()
    #separateTrainingSet()    
    train()


