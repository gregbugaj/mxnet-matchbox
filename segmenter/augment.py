import argparse
import os
import cv2
import numpy as np

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

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
            y_offset = int((l_img.shape[0] - img.shape[0]) / 2)  # Anchored to the upper left
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            img = l_img
            cv2.imwrite(path_dest, img)
        except Exception as e:
            print(e)


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

    args.data_dir_src = "/home/gbugaj/mxnet-training/hicfa/raw/HCFA-AllState"
    args.data_dir_dest = "/home/gbugaj/mxnet-training/hicfa/converted/HCFA-AllState"

    print(args.data_dir_src)
    print(args.data_dir_dest)
    augment(args.data_dir_src, args.data_dir_dest)
    # directory_resize(args.data_dir_src, args.data_dir_dest)
