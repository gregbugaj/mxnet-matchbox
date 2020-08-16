import argparse
import mxnet as mx
from mxnet import nd, gluon
import os
import sys
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2

def val_transform(data):
    data = data.astype('float32')
    augs = [
        mx.image.CenterCropAug((224, 224))
    ]
    for aug in augs:
        data = aug(data)
    # from (H x W x c) to (c x H x W)
    data = mx.nd.transpose(data, (2, 0, 1))
    # Normalzie 0..1 range
    return data.astype('float32') / 255.0


def evaluate():    
    print ('evaluate')
    # load synset for label names
    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    # Use GPU if one exists, else use CPU
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    print(ctx)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = mx.gluon.nn.SymbolBlock.imports("net-symbol.json", ['data'], "net-0000.params", ctx=ctx)
    
    print(net)
    # load an image for prediction
    img = mx.image.imread('../data/flickr/dataset/adidas/4761260517.jpg')
    img = mx.image.imread('../data/flickr/dataset/vodafone/258770610.jpg')
    img = mx.image.imread('../data/flickr/dataset/sprite/2149071419.jpg')
    img = mx.image.imread('../data/flickr/dataset/yahoo/4682534504.jpg')

    print(img.shape)
    # apply transform we did during training
    data = val_transform(img)
    print(data.shape)
    
    # Exand shape into (B x H x W x c)
    data = mx.ndarray.expand_dims(data, axis=0)
    print(data.shape)
    out = net(data.as_in_context(ctx))
    out = mx.nd.SoftmaxActivation(out)
    print(out)
    pred = int(mx.nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()

    print('With prob=%f, %s' %(prob, labels[pred]))
    

if __name__ == '__main__':
    print('Evalute network')
    
    img = mx.random.uniform(0, 255, (300, 300, 3)).astype('uint8')
    print(img.shape)
    img = val_transform(img)
    print(img.shape)
    img = mx.ndarray.expand_dims(img, axis=0)
    print(img.shape)
    
    evaluate()
