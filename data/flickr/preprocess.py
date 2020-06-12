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
from time import time
import matplotlib.pyplot as plt

import logging
import tarfile
logging.basicConfig(level=logging.INFO)

from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.contrib.io import DataLoaderIter



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

        print("Number of Validation Images : {}, {}".format(len(validation_files), validation_files))
        print("Number of Testing Images    : {}, {}".format(len(testing_files), testing_files))
        print("Number of Training Images   : {}, {}".format(len(training_files), training_files))
        
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
        return data, mx.nd.array(([label])).asscalar().astype('float32')

    def val_transform(data, label):        
        data = data.astype('float32')
        augs = [  
            mx.image.CenterCropAug((224,224))
        ]
        for aug in augs:
            data = aug(data)
        # from (H x W x c) to (c x H x W)
        data = mx.nd.transpose(data, (2, 0, 1))
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
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
        X = X.transpose((0,2,3,1)).clip(0,255)/255
        show_images(X, 5, 8)
        break

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    data, label = batch
    return (mx.gluon.utils.split_and_load(data, ctx),
            mx.gluon.utils.split_and_load(label, ctx),
            data.shape[0])
    
def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.nd.array([0])
    n = 0.
    for batch in data_iterator:
        data, label , batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += mx.nd.sum(net(X).argmax(axis = 1) == y).copyto(mx.cpu())
            n +=  y.size
        acc.wait_to_read() # copy from GPU to CPU
    return acc.asscalar() / n

def symbol(num_classes):
    """Define the CNN model"""
    cnn_net = mx.gluon.nn.Sequential()
    with cnn_net.name_scope():
        #  First convolutional layer
        cnn_net.add(mx.gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4,4), activation='relu'))
        cnn_net.add(mx.gluon.nn.MaxPool2D(pool_size=3, strides=2))
        #  Second convolutional layer
        cnn_net.add(mx.gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
        cnn_net.add(mx.gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))
        # Flatten and apply fullly connected layers
        cnn_net.add(mx.gluon.nn.Flatten())
        cnn_net.add(mx.gluon.nn.Dense(4096, activation="relu"))
        cnn_net.add(mx.gluon.nn.Dense(num_classes))
    return cnn_net        

def _train(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix, 
    hybridize=False, learning_rate = 0.01, wd = 0.001):
    """Train model and genereate checkpoints"""
    net.collect_params().reset_ctx(ctx)
    if hybridize == True:
        net.hybridize()
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': wd})

    best_epoch = -1
    best_acc  = 0.0

    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_loss, train_acc, n = 0.0, 0.0, 0.0
        start = time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with mx.autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size

        train_acc = evaluate_accuracy(train_data, net, ctx)
        val_acc = evaluate_accuracy(val_data, net, ctx)
        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Val acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc, val_acc, test_acc, time() - start
        )) 
        if val_acc > best_acc:
            best_acc = val_acc
            if best_epoch!=-1:
                print('Deleting previous checkpoint...')
                os.remove(model_prefix+'-%d.params'%(best_epoch))
            best_epoch = epoch 
            print('Best validation accuracy found. Checkpointing...')
            net.collect_params().save(model_prefix+'-%d.params'%(epoch))      

def train():
    print("Training")    
    # Define the **hyperparameters** for the model
    batch_size = 40
    num_classes = 28
    num_epochs = 2
    num_gpu = 1
    ctx = mx.cpu()#[mx.gpu(i) for i in range(num_gpu)]

    # performs the transformation on the data and returns the updated data set
    root = './trainingset'
    train_dir = os.path.join(root, 'train_data')
    test_dir = os.path.join(root, 'test_data')
    val_dir = os.path.join(root, 'val_data')
    
    # Transform the image when loading using specific transforms
    train_transform, val_transform, test_transform = get_imagenet_transforms(data_shape=224, dtype='float32')

    logging.info("Loading image folder %s, this may take a bit long...", train_dir)    
    train_dataset = ImageFolderDataset(train_dir)
    train_data = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True, last_batch='discard', num_workers=1)

    logging.info("Loading image folder %s, this may take a bit long...", val_dir)    
    val_dataset = ImageFolderDataset(val_dir)
    val_data = DataLoader(val_dataset.transform(val_transform), batch_size,  num_workers=1)

    logging.info("Loading image folder %s, this may take a bit long...", test_dir)    
    test_dataset = ImageFolderDataset(test_dir)
    test_data = DataLoader(test_dataset.transform(test_transform), batch_size,  num_workers=1)

    print("synsets(train_data) : %s " %(train_dataset.synsets))
    print("synsets(val_data) : %s " %(val_dataset.synsets))
    print("synsets(test_data) : %s " %(test_dataset.synsets))

    #display(train_data)
    #display(val_data)
    #display(test_data)

    net = symbol(28)
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    _train(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix='cnn')


if __name__ == "__main__":
    #extractLogosToFolders()
    #separateTrainingSet()    
    train()


