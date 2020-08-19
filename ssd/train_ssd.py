"""Train SSD"""
import argparse
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd

import platform, subprocess, sys, os
import argparse
import cv2
import matplotlib.pyplot as plt

from mxnet import nd

from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.transforms.presets.ssd import SSDDALIPipeline

from mxnet.contrib import amp
from gluoncv import utils as gutils
from gluoncv import data as gdata

from model.vgg_atrous import vgg16_atrous_512
from model.ssd import get_ssd

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx))
    anchors = anchors.as_in_context(mx.cpu())
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_train2017')
        val_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def train(data_dir):
    print("Training SSD : %s" %(data_dir))

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()        

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        prog='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train Single-shot detection network')
    parser.add_argument('--train', default=['classifier', 'detector', 'ssd'], type=str, 
                        help='Train the network [classifier|detector]')
    parser.add_argument('--data-dir', dest='data_dir', 
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data'), type=str)

    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--dataset-root', type=str, default='~/.mxnet/datasets/',
                        help='Path of the directory where the dataset is located.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epochs at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')

    args = parser.parse_args()
    return args
                
if __name__ == "__main__":
    args = parse_args()
    num_classes = 4
    # Causes error
    if args.amp:
        amp.init()

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    
    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name
    
    net = mx.gluon.model_zoo.vision.alexnet(classes=num_classes, ctx = ctx)

    if args.syncbn and len(ctx) > 1:
        net = vgg16_atrous_512(ctx=ctx,
                        norm_kwargs={'num_devices': len(ctx)})
        async_net = vgg16_atrous_512(ctx=ctx)  # used by cpu worker
    else:
        net = vgg16_atrous_512(ctx=ctx)
        async_net = net
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
            # needed for net to be first gpu when using AMP
            for p in net.collect_params().values():
                p.reset_ctx(ctx[0])

    classes = ['ClassA', 'ClassB']
    net = get_ssd('vgg16_atrous', 300, features=vgg16_atrous_512, filters=None,
                  sizes=[30, 60, 111, 162, 213, 264, 315],
                  ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                  steps=[8, 16, 32, 64, 100, 300],
                  classes=classes, dataset='voc'**kwargs)

    print(net)
    net.initialize(mx.init.Normal(), ctx=ctx)
    net.hybridize()
    net(nd.random.normal(shape=(1, 3, 512, 512)))
    net.save_parameters('test.params') 



    if args.train == 'ssd':
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