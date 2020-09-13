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
import os
import time
import numpy
import argparse

from evaluate import recognize, imwrite

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Segmenter evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='data/best.params', type=str)
    parser.add_argument('--dir', dest='dir_src', help='Directory to evaluate', default='data/', type=str)
    parser.add_argument('--output', dest='dir_out', help='Output directory evaluate', default='/tmp/debug', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.network_param = './unet_best.params'
    args.dir_src = '/home/greg/data-hipaa/forms/hcfa-allstate'
    args.dir_out = '/tmp/debug-3'
    
    # /home/gbugaj/mxnet-training/hicfa/raw-2/
    # args.dir_src = '/home/gbugaj/mxnet-training/hicfa/raw/HCFA-Bad-Images'
    args.dir_src = '/home/gbugaj/mxnet-training/hicfa/raw/HCFA-AllState'
    args.dir_src = '/home/gbugaj/mxnet-training/hicfa/raw-2'
    args.dir_src = '/home/greg/data-hipaa/forms/hcfa-allstate'
    
    args.dir_out = '/tmp/debug-2'

    args.debug = False
    ctx = [mx.cpu()]
    
    dir_src = args.dir_src 
    dir_out = args.dir_out 
    network_param = args.network_param

    filenames = os.listdir(dir_src)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for filename in filenames:
        try:
            img_path = os.path.join(dir_src, filename)
            print (img_path)
            src, mask, segment = recognize(network_param, img_path, (3500, 2500), ctx, False)
            # imwrite(os.path.join(dir_out, "%s_%s" % (filename, 'src.tif')), src)
            imwrite(os.path.join(dir_out,'mask', "%s_%s" % (filename, 'mask.tif')), mask)
            imwrite(os.path.join(dir_out, 'segment', "%s_%s" % (filename, 'segment.tif')), segment)
        except Exception as e:
            print(e)
        
        # break



