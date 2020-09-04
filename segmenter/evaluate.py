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
import numpy


# numpy.set_printoptions(threshold=sys.maxsize)

def load_imageXXX(img, width, height):
    im = np.zeros((height, width, 3), dtype='uint8')
    # im[:, :, :] = 128
    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / height
        new_width = int(img.shape[1] / scale)
        diff = (width - new_width) // 2
        img = cv2.resize(img, (new_width, height))

        im[:, diff:diff + new_width, :] = img
    else:
        scale = img.shape[1] / width
        new_height = int(img.shape[0] / scale)
        diff = (height - new_height) // 2

        img = cv2.resize(img, (width, new_height))
        im[diff:diff + new_height, :, :] = img

    data = np.transpose(img, (2, 0, 1))
    # Exand shape into (B x H x W x c)
    data = data.astype('float32')
    data = mx.ndarray.expand_dims(data, axis=0)
    return data


def normalize_image(img):
    rgb_mean = nd.array([0.448, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])
    return (img.astype('float32') / 255.0 - rgb_mean) / rgb_std


def load_image(img, width, height):
    data = np.transpose(img, (2, 0, 1))
    # Expand shape into (B x H x W x c)
    return mx.nd.expand_dims(data, axis=0)


def post_process_mask(label, img_cols, img_rows, n_classes, p=0.5):
    return (np.where(label.asnumpy().reshape(img_cols, img_rows) > p, 1, 0)).astype('uint8')


def post_process_maskA(label, img_cols, img_rows, n_classes, p=0.5):
    pr = label.reshape(n_classes, img_cols, img_rows).transpose([1, 2, 0]).argmax(axis=2)
    return (pr * 255).asnumpy()


if __name__ == '__main__':
    print('Evaluating')
    n_classes = 2
    img_width = 512
    img_height = 512
    ctx = [mx.cpu()]

    net = UNet(channels=3, num_class=n_classes)
    net.load_parameters('./unet_best.params', ctx=ctx)
    # net.load_parameters('./checkpoints/epoch_0357_model.params', ctx=ctx)

    image_path = './data/train/image/5.png'
    img = image.imread(image_path)
    normal = normalize_image(img)

    fig = plt.figure(figsize=(16, 16))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.imshow(img.asnumpy(), cmap=plt.cm.gray)
    ax2.imshow(normal.asnumpy(), cmap=plt.cm.gray)

    # Transform into required BxCxHxW shape
    data = np.transpose(normal, (2, 0, 1))
    # Exand shape into (B x H x W x c)
    data = data.astype('float32')
    data = mx.ndarray.expand_dims(data, axis=0)

    # out = net(data)
    # out = mx.nd.SoftmaxActivation(out)
    # pred = mx.nd.argmax(out, axis=1)

    out = net(data)
    pred = mx.nd.argmax(out, axis=1)
    mask = post_process_mask(pred, 512, 512, 2, p=0.5)

    ax3.imshow(mask, cmap=plt.cm.gray)
    plt.show()
