import argparse
import os
import cv2
import numpy as np
from mxnet import  nd

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
    return resized

def resize_and_frame(image, width, height, color = 255):
    ## Merge two images
    img = image_resize(image, height=height)
    l_img = np.ones((height, height, 3), np.uint8) * color
    s_img = img # np.zeros((512, 512, 3), np.uint8)
    x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
    y_offset = int((l_img.shape[0] - img.shape[0]) / 2)
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

    return l_img

def directory_resize_1024(dir_src, dir_dest):
    print(dir_src)
    print(dir_dest)

    filenames = os.listdir(dir_src)
    filenames.sort()
    if not os.path.exists(dir_dest):
        os.makedirs(dir_dest)
        
    w = 1024
    h = 1024 
    pad = 0

    for filename in filenames:
        try:
            print (filename)
            # open image file
            path = os.path.join(dir_src, filename)
            path_dest = os.path.join(dir_dest, filename) #+ ".png"

            img = cv2.imread(path)     
            img = image_resize(img, height=w)
            img_h, img_w, img_c = img.shape
            print(img.shape)

            ## Merge two images
            l_img = np.ones((h + pad, w + pad, 3), np.uint8) * 255
            s_img = img # np.zeros((512, 512, 3), np.uint8)
            x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
            y_offset = int((l_img.shape[0] - img.shape[0]) / 2)  # Anchored to the upper left
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            img = l_img
            cv2.imwrite(path_dest, img)
        except Exception as e:
            print(e)


def directory_resize(dir_src, dir_dest):
    print(dir_src)
    print(dir_dest)

    filenames = os.listdir(dir_src)
    filenames.sort()
    if not os.path.exists(dir_dest):
        os.makedirs(dir_dest)
        
    w = 1000
    h = 1400 
    pad = 100

    for filename in filenames:
        try:
            print (filename)
            # open image file
            path = os.path.join(dir_src, filename)
            path_dest = os.path.join(dir_dest, filename) + ".png"

            img = cv2.imread(path)     
            img = image_resize(img, width=w)
            img_h, img_w, img_c = img.shape
            print(img.shape)

            ## Merge two images
            l_img = np.ones((h + pad, w + pad, 3), np.uint8) * 255
            s_img = img # np.zeros((512, 512, 3), np.uint8)
            x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
            y_offset = int((l_img.shape[0] - img.shape[0]) / 2)
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            img = l_img
            cv2.imwrite(path_dest, img)
        except Exception as e:
            print(e)

def mean_(dir_src):
    """Calculate mean for all images in directory"""
    print("Calculating Mean and StdDev")

    filenames = os.listdir(dir_src)
    stats = []
    for filename in filenames:
        try:
            # print (filename)
            # open image file
            path = os.path.join(dir_src, filename)
            img = cv2.imread(path, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            mean, std = cv2.meanStdDev(img)
            stats.append(np.array([mean, std]))
            # img = cv2.imread(path)    
            # out = cv2.mean(img)
            # showAndDestroy('normal', img)
            # m = np.mean(img, axis=(0, 1))  / 255.0
            # print(m)
            # avg_color_per_row = np.average(img, axis=0)
            # avg_color = np.average(avg_color_per_row, axis=0) / 255.0
            # print(avg_color)            
        except Exception as e:
            print(e) 

        # break
    vals = np.mean(stats, axis=0) / 255.0
    print(vals)

def augment(dir_src, dir_dest):
    import imgaug as ia
    import imgaug.augmenters as iaa
    print('Augmenting')

    img = cv2.imread('/home/gbugaj/mxnet-training/hicfa/converted/HCFA-AllState/269687_202006290004962_001.tif.png')   
    # images = np.zeros((2, 512, 512, 3), dtype=np.uint8)  # two example images
    # images[:, 64, 64, :] = 255
    images=[img]
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    # polygons
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(10.5, 20.5), (50.5, 30.5), (10.5, 50.5)])
    ], shape=image.shape)
    image_with_polys = psoi.draw_on_image(
        image, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0))
    ia.imshow(image_with_polys)


def showAndDestroy(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 

def augment_image(img, mask, pts, count):
    import random
    import string
    """Augment imag and mask"""
    import imgaug as ia
    import imgaug.augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    def get_random_string(length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str
        
    # add some text to emulate text writing on document
    x1 = pts[0][0][0]
    x2 = pts[1][0][0]
    y1 = pts[0][0][1]
    y2 = pts[-1][0][1]

    tx1 = get_random_string(12)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    upper = int(random.uniform(0, .4) * 10)
    for i in range(upper):
        cv2.putText(img, tx1, (int((x2-x1) // 3 * (1 + random.uniform(1, 2))) , y1 - y1 // (2 + i)), font, random.uniform(.4, .9), (0, 0, 0),2, cv2.LINE_AA)
    seq_shared = iaa.Sequential([
        # sometimes(iaa.Affine(
        #     scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
        #     # shear=(-6, 6),
        #     cval=(0, 0), # Black
        # ))
        iaa.CropAndPad(
            percent=(-0.07, 0.2),
            # pad_mode=ia.ALL,
            pad_mode=["edge"],
            pad_cval=(150, 200)
        )
    ])

    seq = iaa.Sequential([
        sometimes(iaa.SaltAndPepper(0.03, per_channel=False)),
        # Blur each image with varying strength using
        # gaussian blur (sigma between 0 and 3.0),
        # average/uniform blur (kernel size between 2x2 and 7x7)
        # median blur (kernel size between 1x1 and 5x5).
       sometimes(iaa.OneOf([
            iaa.GaussianBlur((0, 2.0)),
            iaa.AverageBlur(k=(2, 7)),
            iaa.MedianBlur(k=(1, 3)),
        ])),

    ], random_order=True)

    masks = []
    images = [] 
    for i in range(count):
        seq_shared_det = seq_shared.to_deterministic()
        image_aug = seq(image = img)
        image_aug = seq_shared_det(image = image_aug)
        mask_aug = seq_shared_det(image = mask)

        masks.append(mask_aug)
        images.append(image_aug)
        # cv2.imwrite('/tmp/imgaug/%s.png' %(i), image_aug)
        # cv2.imwrite('/tmp/imgaug/%s_mask.png' %(i), mask_aug)

    return images, masks

    # image_aug = seq(image = img)
    # showAndDestroy('aug', image_aug)


def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_mask(dir_src, dir_dest, cvat_annotation_file):
    import xml.etree.ElementTree as ET
    import random

    print('resized : %s' % (dir_src))
    print('xml : %s' % (cvat_annotation_file))

    data = {}  
    data['ds'] = []  
    strict = False
    xmlTree = ET.parse(cvat_annotation_file)

    for element in xmlTree.findall("image"):
        name = element.attrib['name']
        points = []
        for polygon_node in element.findall("polygon"):
            points = polygon_node.attrib['points'].split(';')
            break
        size = len(points)
        if strict and size > 0 and size != 4:
            raise ValueError("Expected 4 point got : %s " %(size))
        if size  == 4:
            filename = name.split('/')[-1]
            data['ds'].append({'name': filename, 'points': points}) 

    filenames = os.listdir(dir_src)
    # filenames = random.sample(filenames, len(filenames))
    filenames = dict(zip(filenames, filenames))

    print('Total annotations : %s '% (len(data['ds'])))
    print('Total files : %s '% (len(filenames)))

    target_height = 512

    ensure_exists(os.path.join(dir_dest, 'image'))
    ensure_exists(os.path.join(dir_dest, 'mask'))
    
    for row in data['ds']:
        print(row)
        filename = row['name']
        points = row['points']
        if filename in filenames.keys():
            # open image file
            path = os.path.join(dir_src, filename)
            img = cv2.imread(path) # (1500, 1100, 3) 1400x1000 + 100 padding
            points = [[float(seg) for seg in pt.split(',')] for pt in points]
            # Polygon corner points coordinates 
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            h, w, c = img.shape
            isClosed = True
            color_display = (255, 0, 0) 
            mask_img = np.ones((h, w, c), np.uint8) * 0 # Black canvas

            thickness = 1
            # image = cv2.polylines(img, [pts],  isClosed, color_display, thickness) 
            mask_img = cv2.polylines(mask_img, [pts],  isClosed, (255, 255, 255), thickness) 
            mask_img = cv2.fillPoly(mask_img, [pts], (255, 255, 255) ) # white mask
            # Apply transformations to the image
            aug_images, aug_masks = augment_image(img, mask_img, pts, 10)
            # Add originals
            aug_images.append(img)
            aug_masks.append(mask_img)
            index = 0
            for a_i, a_m in zip(aug_images, aug_masks):
                img = a_i
                mask_img = a_m
                fname = "{}.{}.tif".format(filename.split('.')[0], index)
                # # resize both src and dest
                img_resized = resize_and_frame(img, height=target_height ,width = None, color = 255)
                mask_resized = resize_and_frame(mask_img, height=target_height ,width = None, color = 0)

                path_resized_dest = os.path.join(dir_dest, 'image',  fname)
                path_mask_resized_dest = os.path.join(dir_dest, 'mask', fname)

                print(path_resized_dest)
                cv2.imwrite(path_resized_dest, img_resized)
                cv2.imwrite(path_mask_resized_dest, mask_resized)
                index = index + 1
            break


def split(dir_src, dir_dest):
    import random
    import shutil
    import math

    print('dir_src : %s' % (dir_src))
    print('dir_dest : %s' % (dir_dest))

    # expecting two directories [image, masked]
    image_dir_src = os.path.join(dir_src, 'image')
    mask_dir_src = os.path.join(dir_src, 'mask')

    mask_filenames = os.listdir(mask_dir_src)
    mask_filenames = random.sample(mask_filenames, len(mask_filenames))

    size = len(mask_filenames)

    validation_size = math.ceil(size * 0.1)  # 20 percent validation size
    test_size = math.ceil(size * 0.20)  # 10 percent testing size
    training_size = size - validation_size - test_size  # 70 percent training
    print("Class >>  size = {} training = {} validation = {} test = {} ".format(size, training_size, validation_size, test_size))

    validation_files = mask_filenames[:validation_size]
    testing_files = mask_filenames[validation_size : validation_size+test_size]
    training_files = mask_filenames[validation_size+test_size:]

    print("Number of training images   : {}".format( len(training_files)))
    print("Number of validation images : {}".format(len(validation_files)))
    print("Number of testing images    : {}".format(len(testing_files)))

    # prepare output directories
    test_image_dir_out = os.path.join(dir_dest, 'test', 'image')
    test_mask_dir_out = os.path.join(dir_dest, 'test', 'mask')

    train_image_dir_out = os.path.join(dir_dest, 'train', 'image')
    train_mask_dir_out = os.path.join(dir_dest, 'train', 'mask')

    validation_image_dir_out = os.path.join(dir_dest, 'validation', 'image')
    validation_mask_dir_out = os.path.join(dir_dest, 'validation', 'mask')

    ensure_exists(test_image_dir_out)
    ensure_exists(test_mask_dir_out)
    
    ensure_exists(train_image_dir_out)
    ensure_exists(train_mask_dir_out)

    ensure_exists(validation_image_dir_out)
    ensure_exists(validation_mask_dir_out)

    def copyfiles(files, srcDir, destDir):
        if not os.path.exists(destDir):
            os.makedirs(destDir)

        for filename in files:
            src = os.path.join(srcDir, filename)
            dest = os.path.join(destDir, filename)
            shutil.copy(src, dest)

    copyfiles(training_files, image_dir_src, train_image_dir_out)
    copyfiles(training_files, mask_dir_src, train_mask_dir_out)

    copyfiles(testing_files, image_dir_src, test_image_dir_out)
    copyfiles(testing_files, mask_dir_src, test_mask_dir_out)

    copyfiles(validation_files, image_dir_src, validation_image_dir_out)
    copyfiles(validation_files, mask_dir_src, validation_mask_dir_out)

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Image preprocessor')

    parser.add_argument('--data-src', dest='data_dir_src', 
                        help='data directory to use', default='./data/images', type=str)

    parser.add_argument('--data-dest', dest='data_dir_dest', 
                        help='data directory to output images to', default='./data/out', type=str)

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()

    args.data_dir_src = "/home/greg/data-hipaa/forms/hcfa-allstate"
    args.data_dir_dest = "/home/greg/data-hipaa/forms/converted/hcfa-allstate"

    print(args.data_dir_src)
    print(args.data_dir_dest)
    # augment(args.data_dir_src, args.data_dir_dest)
    # directory_resize(args.data_dir_src, args.data_dir_dest)

    # /home/greg/data-hipaa/forms/converted/hcfa-allstate

    data_dir_src = '/home/greg/data-hipaa/forms/converted/hcfa-allstate'
    data_dir_src = '/home/greg/data-hipaa/forms/converted/raw'
    data_dir_dest = '/home/greg/data-hipaa/forms/converted/resized'
    cvat_annotation_file ='/home/greg/dev/mxnet-matchbox/segmenter/data/annotations/task_hcfa-2020_09_04_21_12_31-cvat for images 1.1/annotations.xml'

    create_mask(data_dir_src, data_dir_dest, cvat_annotation_file)

    data_dir_src = '/home/greg/data-hipaa/forms/converted/resized'
    data_dir_dest = '/home/greg/data-hipaa/forms/splitted'

    # split(data_dir_src, data_dir_dest)

    # mean_('/home/greg/data-hipaa/forms/converted/resized/image')

    data_dir_src = '/home/gbugaj/mxnet-training/hicfa/raw/HCFA-AllState'
    data_dir_dest = '/home/gbugaj/mxnet-training/hicfa/converted_1024'
    # directory_resize_1024(data_dir_src, data_dir_dest)
    
# Class >>  size = 3948 training = 2763 validation = 395 test = 790 
# Number of training images   : 2763
# Number of validation images : 395
# Number of testing images    : 790

# (mxnet-1.7) greg@xpredator:~/dev/mxnet-matchbox/segmenter$ python ./evaluate.py 
# Evaluating
# ratio >> 0.14733812949640288
# out >> (512, 512, 3)
# img >> (512, 512, 3)
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
# Eval time 0.182 sec
# (mxnet-1.7) greg@xpredator:~/dev/mxnet-matchbox/segmenter$ python ./evaluate.py 
# Evaluating
# ratio >> 0.14733812949640288
# out >> (512, 512, 3)
# img >> (512, 512, 3)
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
# Eval time 0.167 sec
