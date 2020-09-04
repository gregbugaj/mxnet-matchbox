import argparse
import os
import cv2


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

def process(dir_src, dir_dest):
    dirs = os.listdir(dir_src)
    dirs.sort()
    print("Classes : {}".format(dirs))

    for clazz in dirs:
        print(clazz)
        clazz_dir = os.path.join(dir_src, clazz)
        clazz_dir_dest = os.path.join(dir_dest, clazz)

        filenames = os.listdir(clazz_dir)
        filenames.sort()
        size = len(filenames)

        print(size)
        print(filenames)
        if not os.path.exists(clazz_dir_dest):
         os.makedirs(clazz_dir_dest)

        for filename in filenames:
            try:
                print (filename)
                # open image file
                path = os.path.join(clazz_dir, filename)
                path_dest = os.path.join(clazz_dir_dest, filename) + ".png"
                # img = cv2.imread(os.path.join(clazz_dir, filename), cv2.IMREAD_GRAYSCALE)            
                w = 1024
                h = 512

                img = cv2.imread(os.path.join(clazz_dir, filename))     
                # img = transform(img)
                # img = image_resize(img, height=h)
                img = image_resize(img, width=w)
                img_h, img_w, img_c = img.shape
                print(img.shape)

                ## Merge two images
                l_img = np.ones((h, w, 3), np.uint8) * 255
                s_img = img # np.zeros((512, 512, 3), np.uint8)
                x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
                y_offset = 0 # Anchored to the upper left
                l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
                img = l_img
                cv2.imwrite(path_dest, img)
            finally:
                print('Error')
    

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Image preprocessor')

    parser.add_argument('--data-src', dest='data_dir_src', 
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data', 'images'), type=str)

    parser.add_argument('--data-dest', dest='data_dir_dest', 
                        help='data directory to output images to', default=os.path.join(os.getcwd(), 'data', 'out'), type=str)

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()

    data_root_dir = "/home/gbugaj/mxnet-training/hicfa/raw/HCFA-AllState"
    data_out_dir = "/home/gbugaj/mxnet-training/hicfa/converted/HCFA-AllState"

    print(args.data_dir_src)
    print(args.data_dir_dest)
    
    process(args.data_dir_src, args.data_dir_dest)
