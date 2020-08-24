import argparse
import os
import cv2
import pandas as pd

def process(dir_src, dir_dest):
    dirs = os.listdir(dir_src)
    dirs.sort()
    print("Classes : {}".format(dirs))

    for clazz in dirs:
        print(clazz)
        clazz_dir = os.path.join(dir_src, clazz)
        clazz_dir_dest = os.path.join(dir_dest, clazz)

        filenames = os.listdir(clazz_dir)
        #random.shuffle(filenames)
        filenames.sort()
        size = len(filenames)

        print(size)
        print(filenames)
        if not os.path.exists(clazz_dir_dest):
         os.makedirs(clazz_dir_dest)

        for filename in filenames:
            # frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
            # frames = [ndimage.imread(frame_path) for frame_path in frames_path]
            # open image file
            img = cv2.imread(os.path.join(clazz_dir, filename))
            # img = cv2.imread(os.path.join(image_source_dir, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # perform transformations on image
            b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
            g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
            r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
            # merge the transformed channels back to an image
            transformed_image = cv2.merge((b, g, r))

            cv2.imshow('Image', transformed_image)
            cv2.waitKey(-1)

            # print (img)
            # cv2.imwrite(imageDestName, crop
        
def processDD(dir_src, data_dir_dest):
    """Preprocess images"""
    print("Preprocessing documentss")
    root_dir = os.getcwd()
    file_list = ['train.csv', 'val.csv']
    image_source_dir = os.path.join(root_dir, 'data/images/')
    data_root = os.path.join(root_dir, 'data')

    for file in file_list:
        
        image_target_dir = os.path.join(data_root, file.split(".")[0])
        
        if not os.path.exists(image_target_dir):
            os.mkdir(image_target_dir)
        
        # read list of image files to process from file
        image_list = pd.read_csv(os.path.join(data_root, file), header=None)[0]
        image_list = image_list[1:]
        print("Start preprocessing images")
        for image in image_list:
            # open image file
            img = cv2.imread(os.path.join(image_source_dir, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # perform transformations on image
            b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
            g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
            r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
            
            # merge the transformed channels back to an image
            transformed_image = cv2.merge((b, g, r))
            target_file = os.path.join(image_target_dir, image)
            print("Writing target file {}".format(target_file))
            cv2.imwrite(target_file, transformed_image)
    print("Finished preprocessing images")
    
def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Image preprocessor')

    parser.add_argument('--data-src', dest='data_dir_src', 
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data', 'images'), type=str)

    parser.add_argument('--data-dest', dest='data_dir_dest', 
                        help='data directory to output images to', default=os.path.join(os.getcwd(), 'data', 'out'), type=str)

    args = parser.parse_args()
    return args
                
if __name__ == "__main__":
    args = parse_args()

    print(args.data_dir_src)
    print(args.data_dir_dest)

    process(args.data_dir_src, args.data_dir_dest)