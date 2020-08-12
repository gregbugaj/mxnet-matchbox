from mxnet import autograd as ag
import mxnet as mx
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.contrib.io import DataLoaderIter
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageFolderDataset
import time
import random
import os
import argparse
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
# python3 -m pip install --upgrade pip
# python3 -m pip install  opencv-python
# python3 -m pip install  python3 -m pip install  matplotlib
# Notes :
# https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/data/datasets.html


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
        X = X.transpose((0, 2, 3, 1)).clip(0, 255)/255
        show_images(X, 5, 8)
        break


def _get_batch_data(batch, ctx):
    """return data, label, batch size on ctx"""
    data, label = batch
    return (mx.gluon.utils.split_and_load(data, ctx),
            mx.gluon.utils.split_and_load(label, ctx),
            data.shape[0])


def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.nd.array([0])
    n = 0.
    for batch in data_iterator:
        data, label, batch_size = _get_batch_data(batch, ctx)
        for X, y in zip(data, label):
            acc += mx.nd.sum(net(X).argmax(axis=1) == y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()  # copy from GPU to CPU
    return acc.asscalar() / n


def get_gluon_network_128(class_numbers):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.Flatten(),
            nn.Dense(units=128, activation="relu"),
            nn.Dense(units=64, activation="relu"),
            nn.Dense(units=class_numbers)
        )
    return net

def get_gluon_network_256(num_classes):
    # construct a MLP
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(num_classes))

    return net

def get_gluon_network_cnn(num_classes):
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


def _train_glueon(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix, hybridize=False, learning_rate=0.1, wd=0.001):
    """Train model and genereate checkpoints"""
    
#   optimizer_params={'learning_rate': 0.1, 'momentum':0.9, 'wd':0.00001}
    optimizer_params={'learning_rate': 0.1, 'momentum':0.9}

    # Initialize network and trainer
    # net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) 
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    # net.collect_params().reset_ctx(ctx)
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)

    # Performance improvement
    if hybridize == True:
        net.hybridize(static_alloc=True, static_shape=True)

    # loss function we will use in our training loop
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    best_epoch = -1
    best_acc = 0.0

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    # Pick a metric
    # metric = mx.metric.Accuracy() # Returns scalars
    metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])  # Returns array

    logger.info("Batch size : s%d" % (batch_size))

    for epoch in range(num_epochs):
        logger.info("Starting Epoch %d" % (epoch))
        tic = time()

        #train_data.reset() # If running as iterator

        btic = time()
        start = time()

        train_loss, train_acc, n = 0.0, 0.0, 0.0
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch_data(batch, ctx)
            outputs = []
            losses = []

            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)  # Forward pass
                    L = loss(z, y)  # Calculate loss
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    losses.append(L)
                    outputs.append(z)

            for l in losses:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in losses])
            n += batch_size
            metric.update(label, outputs) # update the metrics # end of mini-batc

        name, acc = metric.get()

        train_acc = evaluate_accuracy(train_data, net, ctx)
        val_acc = evaluate_accuracy(val_data, net, ctx)
        test_acc = evaluate_accuracy(test_data, net, ctx)

        logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%.6f, %s=%.6f\tTrain acc %.6f, Val acc %.6f, Test acc %.6f' % (
            epoch, i, batch_size/(time()-btic), name[0], acc[0], name[1], acc[1], train_acc, val_acc, test_acc))

        btic = time()

        if val_acc > best_acc:
            best_acc = val_acc
            if best_epoch != -1:
                print('Deleting previous checkpoint...')
                os.remove(model_prefix+'-%d.params' % (best_epoch))
            best_epoch = epoch
            print('Best validation accuracy found. Checkpointing...')
            net.collect_params().save(model_prefix+'-%d.params' % (epoch))

        metric.reset() # end of epoch


def train():
    print("Training")
    mx.random.seed(42)  # Fix the seed for reproducibility
 
    # Define the **hyperparameters** for the model
    batch_size = 16
    num_classes = 28
    num_epochs = 20

    # construct and initialize network.
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

    print(ctx)
    # performs the transformation on the data and returns the updated data set
    root = './trainingset'
    train_dir = os.path.join(root, 'train_data')
    test_dir = os.path.join(root, 'test_data')
    val_dir = os.path.join(root, 'val_data')

    # Transform the image when loading using specific transforms
    train_transform, val_transform, test_transform = get_imagenet_transforms(
        data_shape=224, dtype='float32')

    #logging.info("Loading image folder %s, this may take a bit long...", train_dir)
    train_dataset = ImageFolderDataset(train_dir)
    train_data = DataLoader(train_dataset.transform(
        train_transform), batch_size, shuffle=True, last_batch='discard', num_workers=4)

    #logging.info("Loading image folder %s, this may take a bit long...", val_dir)
    val_dataset = ImageFolderDataset(val_dir)
    val_data = DataLoader(val_dataset.transform(
        val_transform), batch_size,  num_workers=4)

    #logging.info("Loading image folder %s, this may take a bit long...", test_dir)
    test_dataset = ImageFolderDataset(test_dir)
    test_data = DataLoader(test_dataset.transform(
        test_transform), batch_size,  num_workers=4)

    print("synsets(train_data) : %s " % (train_dataset.synsets))
    print("synsets(val_data) : %s " % (val_dataset.synsets))
    print("synsets(test_data) : %s " % (test_dataset.synsets))

    #display(train_data)
    #display(val_data)
    #display(test_data)

    for data, label in train_data:
        print(data.shape, label.shape)
        break

    for i, (x, y) in enumerate(train_data):
        print("index : %s  :: %s x  %s" % (i, x.shape, y.shape))

    net = get_gluon_network_cnn(28)
    _train_glueon(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix='cnn', hybridize=False)

if __name__ == "__main__":
    #extractLogosToFolders()
    #separateTrainingSet()
    train()
