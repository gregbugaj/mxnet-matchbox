import argparse
import os
import cv2
import pandas as pd

from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
import mxnet as mx
import sys
import os
import time

from loader import SegDataset
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
import matplotlib.pyplot as plt

from model_unet import UNet

import logging
import tarfile
logging.basicConfig(level=logging.INFO)
# logging
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('segmenter.log')
logger = logging.getLogger()
logger.addHandler(fh)
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug('\n%s', '-' * 100)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)

os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

# the RGB label of images and the names of lables
COLORMAP = [[0, 0, 0], [255, 255, 255]]
CLASSES = ['background', 'form']

# https://mxnet.apache.org/versions/1.2.1/tutorials/gluon/datasets.html

def _get_batch(batch, ctx, is_even_split=True):
    features, labels = batch
    print('--'*20)
    print(features.shape)
    print(labels.shape)
    print('++'*20)

    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    
    fsplit = gutils.split_and_load(features, ctx, even_split=is_even_split)
    lsplit = gutils.split_and_load(labels, ctx, even_split=is_even_split) 
    print(fsplit[0].shape)
    print(lsplit[0].shape)
    print(features.shape[0])
    print('++'*20)
    
    return fsplit, lsplit, features.shape[0]


def evaluate_accuracy(data_iter, net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels,_ =_get_batch(batch, ctx)
        for x, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(x).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs, log_dir='./', checkpoints_dir='./checkpoints'):
    """Train model and genereate checkpoints"""
    print('Training network : ', num_epochs)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    with open(log_dir + os.sep + 'UNet_log.txt', 'w') as f:
        print('training on', ctx, file=f)
        for epoch in range(num_epochs):
            print('epoch # : ', epoch)
            train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
            for i, batch in enumerate(train_iter):
                print("Batch Index : ", i)
                xs, ys, batch_size = _get_batch(batch, ctx)
                ls = []
                with autograd.record():
                    y_hats = [net(x) for x in xs]
                    ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
                for l in ls:
                    l.backward()
                trainer.step(batch_size)
                train_l_sum += sum([l.sum().asscalar() for l in ls])
                n += sum([l.size for l in ls])
                train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
                m += sum([y.size for y in ys])
            test_acc = evaluate_accuracy(test_iter, net, ctx)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.3f sec'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start), file=f)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.3f sec'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))

            if epoch != 1 and (epoch + 1) % 50 == 0:
                net.save_parameters(checkpoints_dir + os.sep + 'epoch_%04d_model.params' % (epoch + 1))


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Image segmenter')

    parser.add_argument('--data-src', dest='data_dir_src', 
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data', 'images'), type=str)

    parser.add_argument('--data-dest', dest='data_dir_dest', 
                        help='data directory to output images to', default=os.path.join(os.getcwd(), 'data', 'out'), type=str)

    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s 1.0')
    parser.add_argument('--gpu_id',
                        help='a list to enable GPUs. (defalult: %(default)s)',
                        nargs='*',
                        type=int,
                        default=None)
    parser.add_argument('--is_crop',
                        help='a flag to enable cropping. (default: %(default)s)',
                        type=bool,
                        default=True)
    parser.add_argument('--crop_height',
                        help='the height to crop the images. (default: %(default)s)',
                        type=int,
                        default=572)
    parser.add_argument('--crop_width',
                        help='the width to crop the images. (default: %(default)s)',
                        type=int,
                        default=572)
    parser.add_argument('--learning_rate',
                        help='the learning rate of optimizer. (default: %(default)s)',
                        type=float,
                        default=0.1)
    parser.add_argument('--momentum',
                        help='the momentum of optimizer. (default: %(default)s))',
                        type=float,
                        default=0.99)
    parser.add_argument('--batch_size',
                        help='the batch size of model. (default: %(default)s)',
                        type=int,
                        default=3)
    parser.add_argument('--num_epochs',
                        help='the number of epochs to train model. (defalult: %(default)s)',
                        type=int,
                        default=5)
    parser.add_argument('--num_classes',
                        help='the classes of output. (default: %(default)s)',
                        type=int,
                        default=2)
    parser.add_argument('--optimizer',
                        help='the optimizer to optimize the weights of model. (default: %(default)s)',
                        type=str,
                        default='sgd')
    parser.add_argument('--data_dir',
                        help='the directory of datasets. (default: %(default)s)',
                        type=str,
                        default='data')
    parser.add_argument('--log_dir',
                        help="the directory of 'UNet_log.txt'. (default: %(default)s)",
                        type=str,
                        default='./')
    parser.add_argument('--checkpoints_dir',
                        help='the directory of checkpoints. (default: %(default)s)',
                        type=str,
                        default='./checkpoints')
    parser.add_argument('--is_even_split',
                        help='whether or not to even split the data to all GPUs. (default: %(default)s)',
                        type=bool,
                        default=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.gpu_id is None:
        ctx = [mx.cpu()]
    else:
        ctx = [mx.gpu(i) for i in range(len(args.gpu_id))]
        s = ''
        for i in args.gpu_id:
            s += str(i) + ','
        s = s[:-1]
        os.environ['MXNET_CUDA_VISIBLE_DEVICES'] = s
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    # Hyperparameters
    args.num_epochs = 5
    args.batch_size = 16
    args.num_classes = 2
    batch_size = args.batch_size
    num_workers = 2

    root_dir = os.path.join(args.data_dir)

    train_imgs = SegDataset(root='./data/train', colormap=COLORMAP, classes=CLASSES)
    test_imgs = SegDataset(root='./data/test', colormap=COLORMAP, classes=CLASSES)

    train_iter = gdata.DataLoader(train_imgs,batch_size=batch_size,shuffle=True,num_workers=num_workers,last_batch='keep')
    test_iter = gdata.DataLoader(test_imgs,batch_size=batch_size,shuffle=True,num_workers=num_workers,last_batch='keep')
    loss = gloss.SoftmaxCrossEntropyLoss(axis=1)

    if args.optimizer == 'sgd':
        optimizer_params = {'learning_rate': args.learning_rate, 'momentum': args.momentum}
    else:
        optimizer_params = {'learning_rate': args.learning_rate}

    net = UNet(channels = 3, num_class = args.num_classes)
    net.initialize(init=init.Xavier(), ctx=ctx)
    # net.hybridize() # Causes errror with the SHAPE
    # net.initialize(ctx=ctx)
    print(net)

    trainer = gluon.Trainer(net.collect_params(),args.optimizer,optimizer_params)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=args.num_epochs, log_dir=args.log_dir)
    print('Done')