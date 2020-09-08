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

def normalize_image_global(img):
    rgb_mean = nd.array([0.448, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])
    return (img.astype('float32') / 255.0 - rgb_mean) / rgb_std

def normalize_image(img):
    rgb_mean = nd.array([0.94040672, 0.94040672, 0.94040672])
    rgb_std = nd.array([0.14480773, 0.14480773, 0.14480773])
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


def showAndDestroy(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    print('Evaluating')
    payload = np.array([
            [824,154],
            [199,162],
            [206,898],
            [832,888]])

    print(payload)
    a = payload
    ind = np.lexsort((a[:,1],a[:,0]))  
    # out = np.sort(payload)
    print(ind)
    print(a[ind])
    # raise ValueError('done')

    n_classes = 2
    img_width = 1024
    img_height = 1024
    ctx = [mx.cpu()]

    net = UNet(channels=3, num_class=n_classes)
    net.load_parameters('./unet_best.params', ctx=ctx)
    # net.load_parameters('./checkpoints/epoch_0357_model.params', ctx=ctx)
    image_path = './data/validation/image/270205_202006300008659_001.tif.png' # Incorrect aspect ratio
    # image_path = './data/validation/image/269936_202006290009913_001.tif.png' # Incorrect aspect ratio
    image_path = './data/validation/image/270171_202006300007751_001.tif.png'

    name = image_path.split('/')[-1]

    # image_path = './data/train/image/269792_202006290007435_001.tif.png'
    # image_path = '/home/greg/data-hipaa/forms/hcfa-allstate/269689_202006290005163_001.tif'
    
    img = image.imread(image_path)
    normal = normalize_image(img)
    (img_height, img_width, _) = img.shape

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
    mask = post_process_mask(pred, img_width, img_height, 2, p=0.5)

    ax4.imshow(mask, cmap=plt.cm.gray)

    # Extract ROI
    img = cv2.imread(image_path) 
    (cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        print(c)
        print(approx)
        
        if len(approx) == 4:
            # compute the bounding box of the approximated contour and use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            print('{} ,{}, {}, {}'.format(x, y, w, h))
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hull_area)
            print('area = ', area)
            if solidity > .95 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.1):
                # cv2.drawContours(img, [approx], -1, (0, 0, 255), 1)
                res = approx.reshape(-1, 2)
                mask_img = np.ones((img_height, img_width, 3), np.uint8) * 0 # Black canvas
                thickness = 1
                pts = approx
                color_display = (255, 0, 0) 
                x,y,w,h = cv2.boundingRect(c)
                ROI = img[y:y+h, x:x+w]

                # (tl, tr, br, bl) = rect
                max_width = 1024
                max_height = 1024
                # sort by x, y 
                ind = np.lexsort((res[:,1],res[:,0]))  
                res2 = res[ind]

                print(res)
                print(res2)
                # rearange point in in order to get correct perspective
                # pts1 = np.float32([res[0], res[2], res[1],res[3]])
                
                pts1 = np.float32([res[0],res[3], res[1],res[2]])
                pts2 = np.float32([[0, 0], [max_height, 0], [0, max_width], [max_height, max_width]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(img, M, (max_width, max_height))
                ax3.imshow(dst, cmap=plt.cm.gray)

                cv2.imwrite('/tmp/%s'%(name), dst)
                showAndDestroy('Transformed', dst)
    # showAndDestroy('masked', img)
    plt.show()