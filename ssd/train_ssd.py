"""Train SSD"""
import argparse
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
import platform, subprocess, sys, os
import argparse
import cv2
import matplotlib.pyplot as plt

from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet.contrib import amp

import gluoncv as gcv
from gluoncv.utils import download, viz


from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.transforms.presets.ssd import SSDDALIPipeline
from gluoncv.loss import SSDMultiBoxLoss


# from gluoncv.data.batchify import Tuple, Stack, Pad
# from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
# from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
# from gluoncv.data.dataloader import RandomTransformDataLoader

from gluoncv import utils as gutils
from gluoncv import data as gdata
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOCMApMetric, VOC07MApMetric

## notes
# https://gluon-cv.mxnet.io/build/examples_detection/finetune_detection.html


def read_classes():
    """load synset for label names"""
    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]
    return labels        

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

# https://mxnet.incubator.apache.org/versions/1.6/api/python/docs/api/gluon/data/index.html
# https://mxnet.incubator.apache.org/versions/1.6/api/python/docs/_modules/mxnet/gluon/data/dataset.html#RecordFileDataset
# https://gluon-cv.mxnet.io/api/data.datasets.html#gluoncv.data.LstDetection

def get_dataset(dataset_dir, args):
    classes = read_classes()
    num_class = len(classes)
 
    # This two classes are similar but they function differently
    # mx.gluon.data.dataset.RecordFileDataset
    # gcv.data.RecordFileDetection
    train_dataset = gcv.data.RecordFileDetection('./data/hicfa-training/train_data/training.rec', coord_normalized=True)
    test_dataset = gcv.data.RecordFileDetection('./data/hicfa-training/train_data/training.rec', coord_normalized=True)
    # test_dataset = gcv.data.RecordFileDetection('./data/hicfa-training/val_data/validation.rec')
    image, label = train_dataset[0]
    
    print(image.shape, label.shape)
    print('label:', label)

    # display image and label
    ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)
    plt.show()

    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    return train_dataset, test_dataset, val_metric

def dumpRecordFileDetection(record_filename, display_ui, output_to_directory, classes, ctx):    
    """Dump RecordFileDetection to screen or a directory"""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    dataset = gcv.data.RecordFileDetection(record_filename)
    print('images:', len(dataset))
    image, label = dataset[0]
    bboxes=label[:, :4]
    labels=label[:, 4:5]

    print(image.shape, label.shape)
    print('labeldata:', label)
    print('bboxes:', bboxes)
    print('labels:', labels)
 
    image_dump_dir = os.path.join("./dump")
    if not os.path.exists(image_dump_dir):
        os.makedirs(image_dump_dir)

    for i, batch in enumerate(dataset):
        size = len(batch)
        image, label = batch
        print(i, image.shape, label.shape)
        bboxes = label[:, :4]
        labels = label[:, 4:5].astype(np.uint8) 

        if output_to_directory:
            file_path = os.path.join("./dump", "{0}_.png".format(i))
            # Format (c x H x W) 
            img = image.asnumpy().astype(np.uint8) 
            for box, lbl, in zip(bboxes, labels):
                cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(0, 0, 255), 2)
                txt = "{0}".format(classes[lbl[0]])
                cv2.putText(img,txt,(box[0], box[1]), cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1,cv2.LINE_AA, False)
            cv2.imwrite(file_path, img)

        if display_ui:
            ax = viz.plot_bbox(image, bboxes=bboxes, labels=labels, class_names=classes)
            plt.show()

def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""    
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    trainer = gluon.Trainer(
                net.collect_params(), 'sgd',
                {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum},
                update_on_kvstore=(False if args.amp else None))

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                if args.amp:
                    with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)

            local_batch_size = int(args.batch_size)
            ce_metric.update(0, [l * local_batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * local_batch_size for l in box_loss])
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, args.batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)

    print('Completed ')
    # Save the tuned model
    # There are some important distinctions between `net.save_parameters(file_name)` and `net.export(file_name)`
    # https://github.com/apache/incubator-mxnet/blob/master/docs/python_docs/python/tutorials/packages/gluon/blocks/naming.md
    file_name = "net"
    net.export(file_name)
    print('Network saved : %s' % (file_name))

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

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

    parser.add_argument('--no-random-shape', dest='no_random_shape', action='store_true',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                        'and require less memory. However, final model will be slightly worse.')

    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')

    args = parser.parse_args()
    return args
                
if __name__ == "__main__":
    args = parse_args()

    # Causes error
    if args.amp:
        amp.init()

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)
    # args.network = 'ssd_300_resnet50_v1_voc' # ssd_300_resnet50_v1_voc
    args.network = 'resnet50_v1' # ssd_512_resnet50_v1_custom
    args.batch_size = 4
    args.epochs = 20
    args.data_shape = 512
    args.dataset = 'custom'
    
    # args.network = 'ssd_300_vgg16_atrous_voc'
    # construct and initialize network.
    # ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    # ctx = mx.cpu()
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    print(ctx)
    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name
    classes = read_classes()

    print('net_name = %s' %(net_name))
    print('classes  = %s' %(classes))

    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, classes = classes, pretrained_base=False)
        async_net = get_model(net_name,  classes = classes, pretrained_base=False)
    else:
        net = get_model(net_name, classes = classes,  pretrained_base=False)
        async_net = net
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
            # GPU needed for net to be first gpu when using AMP
            for p in net.collect_params().values():
                p.reset_ctx(ctx)
                #p.reset_ctx(ctx[0])

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
        
        # Training 
        # train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
        # train_data, val_data = get_dataloader(async_net, train_dataset, val_dataset, args.data_shape, batch_size, args.num_workers, ctx)
        # train(net, train_data, val_data, eval_metric, ctx, args)

        # Debug 
        # dumpRecordFileDetection('./data/hicfa-training/test_data/test.rec', True, False, classes, ctx)
        # dumpRecordFileDetection('./data/hicfa-training/train_data/training.rec', True, False, classes, ctx)

        # Evaluation
        # https://gluon-cv.mxnet.io/_modules/gluoncv/data/transforms/presets/ssd.html            
        # Does not work on GPU due to 
        # MXNetError: Check failed: inputs[i]->ctx() == default_ctx (gpu(0) vs. cpu(0)) : 
        # CachedOp requires all inputs to live on the same context. But data is on cpu(0) while ssd1_resnetv10_conv0_weight is on gpu(0)

        eval_ssd = True
        if eval_ssd:
            ctx = mx.cpu()
            
            # test_url = 'https://www.signnow.com/preview/100/92/100092626/large.png'
            test_url = 'https://i.pinimg.com/originals/ef/2e/49/ef2e495e4a6c4b8cda30dd3ec9bbacc4.jpg'
            test_url = 'https://www.flaminke.com/wp-content/uploads/2018/09/free-printable-cms-1500-form-02-12-fresh-cms-form-templates-hcfa-best-1500-claim-pdf-template-download-of-free-printable-cms-1500-form-02-12.jpg'
            # test_url = 'https://sc01.alicdn.com/kf/HTB1WEVWIL5TBuNjSspmq6yDRVXaI.jpg'
            
            os.remove('eval.png')
            download(test_url, 'eval.png')
            net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=classes, pretrained_base=False, ctx = ctx)
            net.load_parameters('ssd_512_resnet50_v1_custom_best.params')
            net.collect_params().reset_ctx(ctx)
            
            x, image = gcv.data.transforms.presets.ssd.load_test('eval.png', short = 512)
            class_IDs, scores, bounding_boxes = net(x)

            # print(class_IDs)
            # print(scores)
            # print(bounding_boxes)
            ax = viz.plot_bbox(image, bounding_boxes[0], scores[0],class_IDs[0], class_names=net.classes)
            plt.show()