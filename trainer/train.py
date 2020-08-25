
from mxnet import autograd as ag
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.contrib.io import DataLoaderIter
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageFolderDataset
#from gluoncv.model_zoo import get_model
#from mxnet.gluon.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric

import mxnet as mx
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

from collections import namedtuple
import mxnet as mx
from mxboard import * 

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
# python3 -m pip install mxboard
# python3 -m pip install -r requirements.txt

# Once the images are loaded, we need to ensure the images are of the same size.
# We will resize all the images to be 224 * 224 pixels.
# Let's flip 50% of the training data set horizontally and crop them to 224 * 224 pixels
train_augs = [
    mx.image.HorizontalFlipAug(.5),
    mx.image.RandomCropAug((224, 224))
]

# For the validation and test data sets, let's center crop to get each image to 224 224.
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
        # from (H x W x c) to (c x H x W
        data = mx.nd.transpose(data, (2, 0, 1))
        # Normalzie 0..1 range
        data = data.astype('float32') / 255.0
        # data = mx.nd.image.normalize(data, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # data = mx.nd.image.to_tensor(data) # Fixme : Should be calling this

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
        data = data.astype('float32') / 255.0
        # data = mx.nd.image.normalize(data, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
        # from (B x c x H x W) to (B x H x W x c)
        X = X.transpose((0, 2, 3, 1)).clip(0, 255) / 255
        show_images(X, 5, 8)
        break

def _get_batch_data(batch, ctx):
    """return data, label, batch size on ctx"""
    data, label = batch
    return (mx.gluon.utils.split_and_load(data, ctx, batch_axis=0),
            mx.gluon.utils.split_and_load(label, ctx, batch_axis=0),
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
    net = mx.gluon.nn.HybridSequential()
    #  First convolutional layer
    net.add(mx.gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4,4), activation='relu'))
    net.add(mx.gluon.nn.MaxPool2D(pool_size=3, strides=2))
    #  Second convolutional layer
    net.add(mx.gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
    net.add(mx.gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))
    # Flatten and apply fullly connected layers
    net.add(mx.gluon.nn.Flatten())
    net.add(mx.gluon.nn.Dense(4096, activation="relu"))
    net.add(mx.gluon.nn.Dense(num_classes))

    return net

def _train_gluon(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix, hybridize=False, learning_rate=0.01, wd=0.00001, momentum=0.9):
    """Train model and genereate checkpoints"""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    epochs = num_epochs
    # Data Iterators require call to `reset` during trainging 
    # Initialize network and trainer
    net.initialize(mx.init.Xavier(), ctx=ctx) 
    # net.initialize(mx.init.Normal(), ctx=ctx)

    # net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) # This causes the model to explode with NAN for the loss
    # net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=ctx)
    # net.collect_params().reset_ctx(ctx)

    # trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1E-3})
    # optimizer_params={'learning_rate': 0.01, 'momentum':0.9, 'wd':0.00001}
    # trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)

    # Performance improvement
    if hybridize == True:
        net.hybridize(static_alloc=True, static_shape=True)

    for p in net.collect_params().values():
        p.reset_ctx(ctx)

    # loss function we will use in our training loop
    loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    best_epoch = -1
    best_acc = 0.0

    # Pick a metric
    metric = Accuracy() # Returns scalars
    num_batch = len(train_data)

    lr_factor = 0.75
    # learning rate change at following epochs
    lr_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    # setup learning rate scheduler
    iterations_per_epoch = math.ceil(num_batch)
    # learning rate change at following steps
    lr_steps = [epoch * iterations_per_epoch for epoch in lr_epochs]
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps, factor=lr_factor, base_lr=learning_rate)

    # learning_rate=0.01, wd=0.00001, momentum=0.9
    # setup optimizer with learning rate scheduler, metric, and loss function
    sgd_optimizer = mx.optimizer.SGD(learning_rate=learning_rate, lr_scheduler=schedule, momentum=momentum, wd=wd)
    trainer = gluon.Trainer(net.collect_params(), optimizer=sgd_optimizer)

    # optimizer_params={'learning_rate': 0.01, 'momentum':0.9, 'wd':0.00001}
    # trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)

    # metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])  # Returns array
    logger.info("Batch size : %d" % (batch_size))
    logger.info("Batch total : %d" % (num_batch))

    #training_log = 'logs/train'
    #evaluation_log = 'logs/eval'
    #training_board = mx.contrib.tensorboard.LogMetricsCallback(training_log)

    # define a summary writer that logs data and flushes to the file every 5 seconds
    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    # collect parameter names for logging the gradients of parameters in each epoch
    params = net.collect_params()
    param_names = params.keys()

    # start with epoch 1 for easier learning rate calculation
    global_step = 0
    #for epoch in range(1, epochs + 1):
    for epoch in range(epochs):
        logger.info("Starting Epoch %d" % (epoch))
        tic = time()
        #train_data.reset() # If running as iterator
        train_loss, train_acc, n = 0.0, 0.0, 0.0
        btic = time()

        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch_data(batch, ctx)
            outputs = []
            losses = []

            # FIXME : list to ndarray hangs
            # n = nd.array(data)
            # print(type(n))            

            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)  # Forward pass
                    L = loss_fn(z, y)  # Calculate loss
                    # store the loss and do backward after we have done forward on all GPUs 
                    # for better speed on multiple GPUs.
                    losses.append(L)
                    outputs.append(z)
            for l in losses:
                l.backward()

            entropy = sum([l.mean().asscalar() for l in losses]) / len(losses)
            train_loss += entropy
            sw.add_scalar(tag='cross_entropy', value=entropy, global_step=global_step)
            global_step += 1

            trainer.step(batch_size)
            metric.update(label, outputs) # update the metrics # end of mini-batc
            btic = time()
            print('train_loss: %s' % (train_loss))

            # Log the first batch of images of each epoch
            # FIXME : This is broken values are out of range after they have been normalized
            if i == -1:
                for x, y in zip(data, label):                
                    sw.add_image('first_minibatch', x.reshape((batch_size, 1, 28, 28)), epoch)
    
        if epoch == 0:
            sw.add_graph(net)

        grads = [i.grad() for i in net.collect_params().values()]
        assert len(grads) == len(param_names)
        # logging the gradients of parameters for checking convergence
        for i, name in enumerate(param_names):
            sw.add_histogram(tag=name, values=grads[i], global_step=epoch, bins=1000)


        train_loss /= num_batch
        print('Total train_loss: %s' % (train_loss))
        name, acc = metric.get()

        train_acc = evaluate_accuracy(train_data, net, ctx)
        val_acc = evaluate_accuracy(val_data, net, ctx)
        test_acc = evaluate_accuracy(test_data, net, ctx)
 
        epoch_time = time()-tic
        speed = batch_size/(time()-btic)
        
        logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%.6f, learning-rate: %.6f\tTrain acc %.6f, Val acc %.6f, Test acc %.6f' % (
             epoch, i, speed, name, acc, trainer.learning_rate, train_acc, val_acc, test_acc))

        logger.info('[Epoch %d] time cost: %f'%(epoch, epoch_time))

        # logging training/validation/test accuracy
        sw.add_scalar(tag='accuracy_curves', value=('train_acc', train_acc), global_step=epoch)
        sw.add_scalar(tag='accuracy_curves', value=('valid_acc', val_acc), global_step=epoch)
        sw.add_scalar(tag='accuracy_curves', value=('test_acc', test_acc), global_step=epoch)

        # Training cost
        sw.add_scalar(tag='cost_curves', value=('time', epoch_time), global_step=epoch)
        sw.add_scalar(tag='cost_curves', value=('speed', speed), global_step=epoch)

        btic = time()
        if val_acc > best_acc:
            best_acc = val_acc
            if best_epoch != -1:
                print('Deleting previous checkpoint...')
                fname = os.path.join('params', '%s-%d.params' % (model_prefix, best_epoch))
                if os.path.isfile(fname):
                     os.remove(fname)
                
            best_epoch = epoch
            print('Best validation accuracy found. Checkpointing...')
            fname = os.path.join('params', '%s-%d-%f.params' % (model_prefix, best_epoch, val_acc))
            net.save_parameters(fname)
            logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, val_acc)

        metric.reset() # end of epoch

    # Save the tuned model
    # There are some important distinctions between `net.save_parameters(file_name)` and `net.export(file_name)`
    # https://github.com/apache/incubator-mxnet/blob/master/docs/python_docs/python/tutorials/packages/gluon/blocks/naming.md
    file_name = "net"
    net.export(file_name)
    print('Network saved : %s' % (file_name))

    sw.export_scalars('scalar_dict.json')
    sw.close()


def train(data_dir):
    print("Training")
    mx.random.seed(42)  # Fix the seed for reproducibility

    # Define the **hyperparameters** for the model
    batch_size = 32
    num_classes = 27 + 1 # 27 labeled categories +1 no-classifcation
    num_epochs = 100

    # construct and initialize network.
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    print(ctx)

    # performs the transformation on the data and returns the updated data set
    root = data_dir
    train_dir = os.path.join(root, 'train_data')
    test_dir = os.path.join(root, 'test_data')
    val_dir = os.path.join(root, 'val_data')

    # Transform the image when loading using specific transforms
    train_transform, val_transform, test_transform = get_imagenet_transforms(data_shape=224, dtype='float32')

    #logging.info("Loading image folder %s, this may take a bit long...", train_dir)
    train_dataset = ImageFolderDataset(train_dir)
    train_data = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True, last_batch='discard', num_workers=4)

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
    # net = get_resnet('ResNet50_v2', classes=num_classes, ctx = ctx)
    net = get_gluon_network_cnn(num_classes)

    # Alexnet
    # This requries 'init.normal'
    # net = mx.gluon.model_zoo.vision.alexnet(classes=num_classes, ctx = ctx)

    # Resnet 
    # net = mx.gluon.model_zoo.vision.get_resnet(1, 18, ctx = ctx)
    # net.output = mx.gluon.nn.Dense(num_classes)

    print(net)
    _train_gluon(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix='cnn', hybridize=True)

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
    parser = argparse.ArgumentParser(
        prog='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train Classifier or Single-shot detection network')
    parser.add_argument('--diagnose', default=['mxnet', 'hardware'], type=str, 
                        help='Diagnose mxnet/hardware')
    parser.add_argument('--train', default=['classifier', 'detector'], type=str, 
                        help='Train the network [classifier|detector]')
    parser.add_argument('--data-dir', dest='data_dir', 
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu',  default=False,
                        help='Train on GPU with CUDA')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # im = mx.nd.random_uniform(shape=(3, 10, 10)) * 255
    # im = im.astype('float32') / 255.0
    # print(im)

    #data = mx.nd.transpose(data, (2, 0, 1))
    # Normalzie 0..1 range
    #data = data.astype('float32') / 255.0

    #im = mx.nd.image.to_tensor(im)
#   im_ = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #print (" %s:: %s " %(im_.min(),im_.max()))

    if args.train == 'detector':
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        momentum = args.momentum
        data_dir = os.path.abspath(args.data_dir)
        print(args)
        print(data_dir)
        print('----- Hyperparameters -----')
        print('epochs     : %s' %(epochs))
        print('batch_size : %s' %(batch_size))
        print('lr         : %s' %(lr))
        print('momentum   : %s' %(momentum))
        
        train(data_dir)

    if args.diagnose == 'mxnet':
        check_mxnet()
    
    if args.diagnose == 'hardware':
        check_hardware()
