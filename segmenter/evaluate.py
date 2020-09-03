import mxnet as mx
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

def v1():
    # ctx = mx.cpu()gpu
    ctx = mx.cpu()
    net = UNet(channels = 3, num_class = 2)
    net.load_parameters('./checkpoints/epoch_0020_model.params', ctx=ctx)
    print(net)
    image_path = './data/train/image/3.png'
    # load an image for prediction
    
    img = mx.image.imread(image_path)
    org = cv2.imread(image_path)
    print(img.shape)
    # Convert HxWxC > CxHxW
    # img = normalize_image(img)
    img = np.transpose(img, (2, 0, 1))
    # Exand shape into (B x H x W x c)
    data = mx.ndarray.expand_dims(img, axis=0)
    data = data.astype('float32')
    data = data.as_in_context(ctx)
    # Expecting data in (batch_size, channels, height, width)
    out = net(data)
    predictions = nd.argmax(out, axis=1)
    pred = predictions.asnumpy()
    print('Model predictions: ', pred)
    p2 = np.transpose(predictions, (1,2,0))   # C X H X W  ==>   H X W X C

    plt.imshow(p2.asnumpy())
    plt.show()

    cv2.imwrite('./data/out.png', p2.asnumpy())

def load_imageXXX(img, width, height):
    im = np.zeros((height, width, 3), dtype='uint8')
    #im[:, :, :] = 128
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
    print(type(img))
    rgb_mean = nd.array([0.448, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])

    # raw = (img.astype('float32') / 255.0  - .448) / .229
    raw = (img.astype('float32') / 255.0 - rgb_mean) /rgb_std
    return mx.nd.array(raw)
    # return (img.astype('float32') / 255.0 - rgb_mean) /rgb_std

def load_image(img, width, height):
    data = np.transpose(img, (2, 0, 1))
    data = mx.nd.array(data) # from numpy.ndarray to mxnet ndarray.
    # Expand shape into (B x H x W x c)
    data = data.astype('float32')
    return mx.ndarray.expand_dims(data, axis=0)

def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

def v2():
    raise ValueError("done")
    train_imgs = SegDataset(root='./data/train', colormap=COLORMAP, classes=CLASSES)
    train_iter = gdata.DataLoader(train_imgs,batch_size=batch_size,shuffle=True,num_workers=num_workers,last_batch='keep')
    for i, batch in enumerate(train_iter):
        features, labels = batch
        _features = gutils.split_and_load(features, ctx, even_split=True)
        print('i = ', i)
        for x in _features:
            print('evaluating')
            out = net(x)
            print('output')
            print(out)
 
            # mask = post_process_mask(output[0], 512, 512, 2, .05)
            # print(mask)

            # cv2.imshow('mask', mask)
            # cv2.waitKey()
        break
    print("Batch complete")

def post_process_mask(label, img_cols, img_rows, n_classes, p=0.5):
    return (np.where(label.asnumpy().reshape(img_cols, img_rows) > p, 1, 0)).astype('uint8')

def post_process_maskA(label, img_cols, img_rows, n_classes, p=0.5):
    pr = label.reshape(n_classes, img_cols, img_rows).transpose([1,2,0]).argmax(axis=2)
    return (pr*255).asnumpy()

if __name__ == '__main__':
    print('Evaluating')
    n_classes = 2
    img_width = 512
    img_height = 512
    ctx = [mx.cpu()]

    net = UNet(channels = 3, num_class = n_classes)
    net.load_parameters('./checkpoints/epoch_0010_model.params', ctx=ctx)

    image_path = './data/train/image/5.png'
    # a = mx.nd.normal(shape=(1, 3, 512, 512))
    # logits = net(a)
    # print(logits.shape)

    # # load an image for prediction
    testimg = cv2.imread(image_path, 1)
    imgX = load_image(testimg, img_width, img_height)
    print(imgX.shape)
    # cv2.imshow('test', testimg)
    # cv2.waitKey()

    normal = normalize_image(testimg)
    # img = mx.random.uniform(0, 255, (512, 512, 3)).astype('uint8')
    img = np.transpose(normal, (2, 0, 1))
    img = mx.ndarray.expand_dims(img, axis=0)
    data = img #imgX.astype(np.float32)

    # width = 12
    # height = 12
    # plt.figure(figsize=(width, height))
    # plt.subplot(121)
    # plt.imshow(normal, cmap=plt.cm.gray)
    # plt.show()

    # plt.subplot(122)
    # plt.imshow(post_process(batch.data[0][idx]).asnumpy()*post_process_mask(outputs[idx]), cmap=plt.cm.gray)
    #out = net(data).argmax(axis=1)
    out = net(data)
    out = mx.nd.SoftmaxActivation(out)
    pred = mx.nd.argmax(out, axis=1)
    mask = post_process_mask(pred, 512, 512, 2, p = 0.5)

  #  pred = int(mx.nd.argmax(out, axis=1).asscalar())
   # prob = out[0][pred].asscalar()
    #print(out.shape)
    #mask = post_process_maskA(out, 512, 512, 2, p = 0.5)
    #print(mask.shape)


    # ost_process_mask(label, img_cols, img_rows, n_classes, p=0.5):
    # mask = post_process_mask(out[0][1], 512, 512, 2, p = 0.5)
    # mask = post_process_mask(out[1], 512, 512, p = 0.5)

    # cv2.imshow('test', out2)
    # cv2.waitKey()

    plt.imshow(mask, cmap=plt.cm.gray)
    plt.show()
 
  