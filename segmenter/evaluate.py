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

# numpy.set_printoptions(threshold=sys.maxsize)

def normalize_image(img):
    """normalize image for bitonal processing"""
    # rgb_mean = nd.array([0.94040672, 0.94040672, 0.94040672])
    # rgb_std = nd.array([0.14480773, 0.14480773, 0.14480773])
    # second augmented set
    rgb_mean = nd.array([0.93610591, 0.93610591, 0.93610591])
    rgb_std = nd.array([0.1319155, 0.1319155, 0.1319155])
    return (img.astype('float32') / 255.0 - rgb_mean) / rgb_std

def post_process_mask(pred, img_cols, img_rows, n_classes, p=0.5):
    """ 
    pred is of type mxnet.ndarray.ndarray.NDArray
    so we are converting it into numpy
    """
    return (np.where(pred.asnumpy().reshape(img_cols, img_rows) > p, 1, 0)).astype('uint8')

def showAndDestroy(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 

def find_best_point(point, point_candidates, max_distance = sys.maxsize):
    """Find best point by performing L2 norm (eucledian distance) calculation"""
    best = None
    for p in point_candidates:
        dist = numpy.linalg.norm(point - p)
        if dist <= max_distance:
            max_distance = dist
            best = p
    return max_distance, best

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return r, resized

def resize_and_frame(image, width = None, height = None, color = 255):
    ## Merge two images
    ratio, img = image_resize(image, height=height)
    l_img = np.ones((height, height, 3), np.uint8) * color
    s_img = img # np.zeros((512, 512, 3), np.uint8)
    x_offset = 0 # int((l_img.shape[1] - img.shape[1]) / 2) 
    y_offset = 0 # int((l_img.shape[0] - img.shape[0]) / 2)
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

    return ratio, l_img

def recognize(network_parameters, image_path, form_shape, ctx, debug):
    """Recognize form

    *network_parameters* is a filename for trained network parameters,
    *image_path* is an filename to the image path we want to evaluate.
    *form_shape* is the shape that the output form will be translated into
    *ctx* is the mxnet context we evaluating on
    *debug* this flag idicates if we are goig to show debug information

    Algorithm :
        Setup recogintion network(Modified UNET)
        Prepare images
        Run prediction on the network
        Reshape predition onto target image

    Return an tupple of src, mask, segment
    """

    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    start = time.time()

    # At one point this can be generalized but right now I don't see this changing 
    n_classes = 2
    n_channels = 3
    img_width = 512
    img_height = 512

    # Setup network
    net = UNet(channels = n_channels, num_class = n_classes)
    net.load_parameters(network_parameters, ctx=ctx)
    
    # Srepare images
    src = cv2.imread(image_path) 
    ratio, resized_img = resize_and_frame(src, height=512)
    img = mx.nd.array(resized_img)
    normal = normalize_image(img)
    name = image_path.split('/')[-1]

    if debug:
        fig = plt.figure(figsize=(16, 16))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.imshow(resized_img, cmap=plt.cm.gray)
        ax2.imshow(normal.asnumpy(), cmap=plt.cm.gray)

    # Transform into required BxCxHxW shape
    data = np.transpose(normal, (2, 0, 1))
    # Exand shape into (B x H x W x c)
    data = data.astype('float32')
    data = mx.ndarray.expand_dims(data, axis=0)
    # prediction 
    out = net(data)
    pred = mx.nd.argmax(out, axis=1)
    nd.waitall() # Wait for all operations to finish as they are running asynchronously
    mask = post_process_mask(pred, img_width, img_height, n_classes, p=0.5)

    rescaled_height = int(img_height / ratio)
    ratio, rescaled_mask = image_resize(mask, height=rescaled_height)  
    mask = rescaled_mask
    if debug:
        ax4.imshow(mask, cmap=plt.cm.gray) 
    # Extract ROI
    (cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Expected size of the new image
    height = form_shape[0] 
    width = form_shape[1]
    cols = width + 1
    rows = height + 1
    segment = None

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        if len(approx) == 4:
            # compute the bounding box of the approximated contour and use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hull_area)

            if area > 50000 and solidity > .95 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.4):
                # cv2.drawContours(img, [approx], -1, (0, 0, 255), 1)
                # Keypoint order : (top-left, top-right, bottom-right, bottom-left)                
                # Rearange points in in order to get correct perspective change
                res = approx.reshape(-1, 2)
                x,y,w,h = cv2.boundingRect(c)
                _, top_left = find_best_point([y, x], res)
                _, top_right = find_best_point([y, x + w], res)
                _, bottom_right = find_best_point([y+h, x + w], res)
                _, bottom_left = find_best_point([y+h, x], res)
                src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
                dst_pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                segment = cv2.warpPerspective(src, M, (cols, rows), flags = cv2.INTER_LINEAR)
                
                if debug:
                    ax3.imshow(segment, cmap=plt.cm.gray)
                    cv2.imwrite('/tmp/%s'%(name), segment)
                    showAndDestroy('Extracted Segment', segment)

    dt = time.time() - start
    print('Eval time %.3f sec' % (dt))
    if debug:
        plt.show()

    mask = mask * 255 # currently mask is 0 / 1 
    return src, mask, segment

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Segmenter evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='data/input.png', type=str)
    parser.add_argument('--image', dest='img_path', help='Image filename to evaluate', default='data/input.png', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args()

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

if __name__ == '__main__':
    args = parse_args()
    args.network_param = './unet_best.params'
    args.img_path = '/home/gbugaj/mxnet-training/hicfa/raw/HCFA-AllState/269792_202006290007435_001.tif'
    # args.img_path = '/home/greg/data-hipaa/forms/hcfa-allstate/269687_202006290004962_001.tif'

    args.debug = True

    ctx = [mx.cpu()]
    src, mask, segment = recognize(args.network_param, args.img_path, (3500, 2500), ctx, args.debug)
    name = args.img_path.split('/')[-1]

    imwrite('/tmp/debug/%s_src.tif' % (name), src)
    imwrite('/tmp/debug/%s_mask.tif' % (name), mask )
    imwrite('/tmp/debug/%s_segment.tif' % (name), segment)

