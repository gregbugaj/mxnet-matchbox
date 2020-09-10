import mxnet as mx
from mxnet import image, nd
import numpy as np
from mxnet import nd, autograd
from mxnet.gluon import data as gdata
from model_unet import UNet
from loader import SegDataset
import cv2
import matplotlib.pyplot as plt

from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
import sys
import time
import numpy
import argparse

from evaluate import recognize

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Segmenter evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='data/input.png', type=str)
    parser.add_argument('--image', dest='img_path', help='Image filename to evaluate', default='data/input.png', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.network_param = './unet_best.params'
    args.img_path = '/home/gbugaj/mxnet-training/hicfa/raw/HCFA-Bad-Images/2020043001127-1.TIF'
    args.debug = True

    ctx = [mx.cpu()]
    src, mask, segment = recognize(args.network_param, args.img_path, (3500, 2500), ctx, args.debug)
    name = args.img_path.split('/')[-1]
    cv2.imwrite('/tmp/debug/%s_src.png' % (name), src)
    cv2.imwrite('/tmp/debug/%s_mask.png' % (name), mask)
    cv2.imwrite('/tmp/debug/%s_segment.png' % (name), segment)

