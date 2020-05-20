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
        name = parts[0]
        label = parts[1].lower()    
        clazz = int(parts[2])
        x1 = int(parts[3])
        y1 = int(parts[4])
        x2 = int(parts[5])
        y2 = int(parts[6])
        w = x2 - x1
        h = y2 - y1

        imagePath = os.path.join("./flickr_logos_27_dataset_images", name)
        labelDir = os.path.join("./dataset", label)
        imageDestName = os.path.join("./dataset", label, name)
        if not os.path.exists(labelDir):
            os.makedirs(labelDir)

        print (labelDir)
        img = cv2.imread(imagePath)
        crop = img[y1:y1+h, x1:x1+w]
        #cv2.imshow('Image', crop)
        #cv2.waitKey(100)
        print(imageDestName)
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

def transform(srcImage):
    print("image : {}".format(srcImage))
    img = cv2.imread(srcImage)
    cv2.imshow('Image', img)
    cv2.waitKey(10000)
    
if __name__ == "__main__":
#    extractLogosToFolders()
    #separateTrainingSet()
    transform("./trainingset/test_data/yahoo/3002734592.jpg")


