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
        image, _ = mx.image.random_size_crop(
            image, (data_shape, data_shape), 0.08, (3/4., 4/3.))
        image = mx.nd.image.random_flip_left_right(image)
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(
            0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
    """return data and label on ctx"""
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


def lenet5(num_classes):
    """LeNet-5 Symbol"""
    #pylint: disable=no-member
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max",
                           kernel=(2, 2), stride=(2, 2))
    # second conv
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max",
                           kernel=(2, 2), stride=(2, 2))
    # first fullc
    flatten = mx.sym.Flatten(data=pool2)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=num_classes)
    # loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    #pylint: enable=no-member
    return lenet


def get_alexnet(num_classes):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(data=input_data, kernel=(
        11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    lrn1 = mx.symbol.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    pool1 = mx.symbol.Pooling(
        data=lrn1, pool_type="max", kernel=(3, 3), stride=(2, 2))
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    lrn2 = mx.symbol.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    pool2 = mx.symbol.Pooling(data=lrn2, kernel=(
        3, 3), stride=(2, 2), pool_type="max")
    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(
        3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)

    # stage 6
    fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax


def cnn_net_3XX(num_classes):
    """Define the AlexNet CNN model"""
    data = mx.symbol.Variable('data')
    conv1 = mx.sym.Convolution(data=data, pad=(
        1, 1), kernel=(3, 3), num_filter=24, name="conv1")
    relu1 = mx.sym.Activation(data=conv1, act_type="relu", name="relu1")
    pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(
        2, 2), stride=(2, 2), name="max_pool1")
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(
        3, 3), num_filter=48, name="conv2", pad=(1, 1))
    relu2 = mx.sym.Activation(data=conv2, act_type="relu", name="relu2")
    pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(
        2, 2), stride=(2, 2), name="max_pool2")

    conv3 = mx.sym.Convolution(data=pool2, kernel=(
        5, 5), num_filter=64, name="conv3")
    relu3 = mx.sym.Activation(data=conv3, act_type="relu", name="relu3")
    pool3 = mx.sym.Pooling(data=relu3, pool_type="max", kernel=(
        2, 2), stride=(2, 2), name="max_pool3")

    #conv4 = mx.sym.Convolution(data=conv3, kernel=(5,5), num_filter=64, name="conv4")
    #relu4 = mx.sym.Activation(data=conv4, act_type="relu", name="relu4")
    #pool4 = mx.sym.Pooling(data=relu4, pool_type="max", kernel=(2,2), stride=(2,2),name="max_pool4")

    # first fullc layer
    flatten = mx.sym.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500, name="fc1")
    relu3 = mx.sym.Activation(data=fc1, act_type="relu", name="relu3")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=43, name="final_fc")
    # softmax loss
    mynet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

    return mynet


def _train_symbolic(net, ctx, train_data: DataLoader, val_data: DataLoader, test_data: DataLoader, batch_size, num_epochs, model_prefix,
                    hybridize=False, learning_rate=0.1, wd=0.001):
    """Train model and genereate checkpoints"""
    #create adam optimiser
    print("Training datatypes")
    print(type(net))
    print(type(train_data))
    print(type(val_data))
    print(type(test_data))

    best_epoch = -1
    best_acc = 0.0

    train_iter = DataLoaderIter(train_data)
    # train_iter=SimpleIter(gluon_data_loader)

    # create module
    #mod = mx.mod.Module(symbol=net, data_names=['data'], label_names=['softmax_label'])
    mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])
    # allocate memory by given the input data and lable shapes
    mod.bind(data_shapes=train_iter.provide_data,
             label_shapes=train_iter.provide_label)

    # initialize parameters by uniform random numbers
    initializer = mx.initializer.Normal()
    #mod.init_params(initializer=init)
    #optimizer = 'adam'

    """ if optimizer == 'sgd':
        # use Sparse SGD with learning rate 0.1 to train
        sgd = mx.optimizer.SGD(momentum=0.1, clip_gradient=5.0, learning_rate=0.01,
                                rescale_grad=1.0/batch_size)
        mod.init_optimizer(optimizer=sgd)
        if num_epochs is None:
            num_epochs = 10
        expected_accuracy = 0.02
    elif optimizer == 'adam':
        # use Sparse Adam to train
        adam = mx.optimizer.Adam(clip_gradient=5.0, learning_rate=0.0005,
                                    rescale_grad=1.0/batch_size)
        mod.init_optimizer(optimizer=adam)
        if num_epochs is None:
            num_epochs = 10
        expected_accuracy = 0.05
    elif optimizer == 'adagrad':
        # use Sparse AdaGrad with learning rate 0.1 to train
        adagrad = mx.optimizer.AdaGrad(clip_gradient=5.0, learning_rate=0.01,
                                        rescale_grad=1.0/batch_size)
        mod.init_optimizer(optimizer=adagrad)
        if num_epochs is None:
            num_epochs = 20
        expected_accuracy = 0.09
    else:
        raise AssertionError("Unsupported optimizer type '" + optimizer + "' specified")  """

    # use accuracy as the metric
    metric = mx.metric.create('MSE')

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    eval_metrics = ['accuracy']

    #batch_end_callback = mx.callback.Speedometer(args.batch_size, 50)
    _batch_end_callback = mx.callback.Speedometer(
        batch_size, batch_size, auto_reset=False),

    def checkpoint(arg1, arg2, arg3, arg4):
        print("Checkpoint now %s" % (type(arg1)))
    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    optimizer_params = {
        'learning_rate': learning_rate,
        'wd': wd,
        #    'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # fit the module
    mod.fit(train_iter,
            eval_data=train_iter,
            optimizer='sgd',
            optimizer_params=optimizer_params,
            initializer=initializer,
            eval_metric=eval_metrics,
            num_epoch=num_epochs,
            batch_end_callback=_batch_end_callback,
            epoch_end_callback=checkpoint
            )


""" 
    for epoch in range(num_epochs):
        print("Starting Epoch %d"  % (epoch))

        train_loss, train_acc, n = 0.0, 0.0, 0.0
        start = time()  
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch_data(batch, ctx)
           # print(" Batch [%d] "  % (i))
 """


def get_net(class_numbers):
    # construct a MLP
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(class_numbers))
    # initialize the parameters
    net.collect_params().initialize()
    return net


def get_symbol_mlp(num_classes, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp



def _train_glueon(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix,
                  hybridize=False, learning_rate=0.1, wd=0.001):
    """Train model and genereate checkpoints"""
    net.collect_params().reset_ctx(ctx)
    if hybridize == True:
        net.hybridize()

    optimizer_params={'learning_rate': 0.1, 'momentum':0.9, 'wd':0.00001}
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)

    best_epoch = -1
    best_acc = 0.0

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    # Pick a metric
    # metric = mx.metric.Accuracy() # Returns scalars
    metric = CompositeEvalMetric(
        [Accuracy(), TopKAccuracy(5)])  # Returns array

    for epoch in range(num_epochs):
        logger.info("Starting Epoch %d" % (epoch))
        tic = time()

        #train_data.reset() # If running as iteratiro

        metric.reset()
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
                ag.backward(losses)

            metric.update(label, outputs)
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in losses])
            n += batch_size

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


def train():
    print("Training")
    mx.random.seed(42)  # Fix the seed for reproducibility

    # Define the **hyperparameters** for the model
    batch_size = 32
    num_classes = 28
    num_epochs = 10

    # construct and initialize network.
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

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
        train_transform), batch_size, shuffle=True, last_batch='discard', num_workers=1)

    #logging.info("Loading image folder %s, this may take a bit long...", val_dir)
    val_dataset = ImageFolderDataset(val_dir)
    val_data = DataLoader(val_dataset.transform(
        val_transform), batch_size,  num_workers=1)

    #logging.info("Loading image folder %s, this may take a bit long...", test_dir)
    test_dataset = ImageFolderDataset(test_dir)
    test_data = DataLoader(test_dataset.transform(
        test_transform), batch_size,  num_workers=1)

    print("synsets(train_data) : %s " % (train_dataset.synsets))
    print("synsets(val_data) : %s " % (val_dataset.synsets))
    print("synsets(test_data) : %s " % (test_dataset.synsets))

    #display(train_data)
    #display(val_data)
    #display(test_data)

    # net = cnn_net_3(28)
    net = get_alexnet(28)

    for data, label in train_data:
        print(data.shape, label.shape)
        break

    for i, (x, y) in enumerate(train_data):
        print("index : %s  :: %s x  %s" % (i, x.shape, y.shape))
 
    net = get_net(28)
    #symbol = get_symbol_mlp(28)
    #_train(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix='cnn')
    _train_glueon(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix='cnn', hybridize=True)

if __name__ == "__main__":
    #extractLogosToFolders()
    #separateTrainingSet()
    train()
