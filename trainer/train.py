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
import platform, subprocess, sys, os
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
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    # Data Iterators require call to `reset` during trainging 

    # train_data = DataLoaderIter(train_dataXX)
    optimizer_params={'learning_rate': 0.01, 'momentum':0.9, 'wd':0.00001}
    # Initialize network and trainer
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) 
    # net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) # This causes the model to explode with NAN for the loss
    # net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=ctx)
    net.collect_params().reset_ctx(ctx)

    # net.collect_params().reset_ctx(ctx)
    # trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1E-3})
    # trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)

    # Performance improvement
    if hybridize == True:
        net.hybridize(static_alloc=True, static_shape=True)

    # loss function we will use in our training loop
    loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    best_epoch = -1
    best_acc = 0.0

    # Pick a metric
    # metric = mx.metric.Accuracy() # Returns scalars
    metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])  # Returns array
    logger.info("Batch size : s%d" % (batch_size))

    for epoch in range(num_epochs):
        logger.info("Starting Epoch %d" % (epoch))
        tic = time()
#        train_data.reset()
        #train_data.reset() # If running as iterator
        btic = time()
        start = time()

        train_loss, train_acc, n = 0.0, 0.0, 0.0
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch_data(batch, ctx)
            # print('batch.data[0] : %s' % (batch.data[0].shape[0]))
            outputs = []
            losses = []

            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)  # Forward pass
                    L = loss_fn(z, y)  # Calculate loss
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    losses.append(L)
                    outputs.append(z)
                    print('   loss[L] : %s' % (L))

            for l in losses:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in losses])

            n += batch_size
            metric.update(label, outputs) # update the metrics # end of mini-batc
            print('train_loss: %s' % (train_loss))

        print('Total train_loss: %s' % (train_loss))
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
    batch_size = 64
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
    train_transform, val_transform, test_transform = get_imagenet_transforms(data_shape=224, dtype='float32')

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

    # Get the model ResNet50_v2
    net = get_model('ResNet50_v2', classes=num_classes, ctx = ctx)
    #net = get_gluon_network_cnn(num_classes)
    _train_glueon(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix='cnn', hybridize=False)


def get_build_features_str():
    import mxnet.runtime
    features = mxnet.runtime.Features()
    return '\n'.join(map(str, list(features.values())))

def check_mxnet():
    print('----------MXNet Info-----------')
    try:
        import mxnet
        print('Version      :', mxnet.__version__)
        ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
        gpu = True if mx.context.num_gpus() else False
        print('GPU Detected :', gpu)
        print('Context      :', ctx)

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


def check_hardware():
    print('----------Hardware Info----------')
    print('machine      :', platform.machine())
    print('processor    :', platform.processor())
    if sys.platform.startswith('darwin'):
        pipe = subprocess.Popen(('sysctl', '-a'), stdout=subprocess.PIPE)
        output = pipe.communicate()[0]
        for line in output.split(b'\n'):
            if b'brand_string' in line or b'features' in line:
                print(line.strip())
    elif sys.platform.startswith('linux'):
        subprocess.call(['lscpu'])
    elif sys.platform.startswith('win32'):
        subprocess.call(['wmic', 'cpu', 'get', 'name'])

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Matchbox trainging system')
        
    #parser.add_argument('--region', default='', type=str, help="Additional sites in which region(s) to test. Specify 'cn' for example to test mirror sites in China.")
    parser.add_argument('--diagnose', default='mxnet', type=str, help='Diagnose mxnet/hardware')
    parser.add_argument('--train', default='', type=str, help='Train the network')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.train:
        train()

    if args.diagnose == 'mxnet':
        check_mxnet()
    
    if args.diagnose == 'hardware':
        check_hardware()
    
    