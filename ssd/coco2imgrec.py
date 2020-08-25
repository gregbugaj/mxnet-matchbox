from mxnet import autograd as ag
import mxnet as mx
from mxnet import image  

import time
import random
import os
import argparse
import json  
from skimage import io  

## python -m pip install scikit-image
## https://github.com/leocvml/mxnet-im2rec_tutorial
## https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html

## Steps 
## 1. Convert
## 2. python im2rec.py --pack-label training.lst ownSet

# python /home/greg/dev/3rdparty/mxnet/tools/im2rec.py
# python /home/greg/dev/3rdparty/mxnet/tools/im2rec.py --pack-label ./data/hicfa-training/test_data/test.lst ownSet

def convert(coco_filename, lst_filename):
    print("Converting")   
    root_dir = os.path.dirname(coco_filename) 
    print(coco_filename)   
    print(lst_filename)   
    print(root_dir)   

    with open(coco_filename, 'r') as f:  
        DataSets = json.load(f)  
    
    print(DataSets['annotations'][0])  
    print(DataSets['images'][0])  

    ## save class and own dataset .json  
    jsonName = os.path.join(root_dir, 'ownset.json') 
    data = {}  
    data['DataSet'] = []  
    with open(jsonName, 'w') as outfile:  
        for DataSet in DataSets['annotations']:  
            box = DataSet['bbox']  
            img_id = str(DataSet['image_id'])  
            coco_name = ""
            for im in DataSets['images']:                
                if str(im['id']) == str(img_id):
                    coco_name =  im['file_name']
                    break
            
            img_name = coco_name
            coco_filename = os.path.join(root_dir, coco_name)

            with open(coco_filename, 'rb') as f:  
                img = image.imdecode(f.read())  
                height = img.shape[0]  
                width  = img.shape[1]  
                box[0] = box[0]/width  #normalize
                box[2] = box[2]/width  
                box[1] = box[1]/height  
                box[3] = box[3]/height  

            # io.imsave(directory + img_name, img.asnumpy())  
            data['DataSet'].append({  
                'img_name': img_name,  
                'height': height,  
                'width': width,  
                'bbox': box,  
                'class':DataSet['category_id']  
            })  

        json.dump(data, outfile)  
    print('save ok')  
    print('Step 2 : Create LST filename')  

    with open(jsonName, 'r') as f:  
        DataSet = json.load(f)  
    
    print(DataSet['DataSet'][0]['img_name'])  
    
    img_idx = 0  
    with open(lst_filename, 'w+') as f:  
        for Data in DataSet['DataSet']:  
    
            x_min = Data['bbox'][0]  
            y_min = Data['bbox'][1]  
            x_max = Data['bbox'][0]+ Data['bbox'][2]  
            y_max = Data['bbox'][1]+ Data['bbox'][3] 
            f.write(  
                    str(img_idx) + '\t' +  # idx  
                    str(4) + '\t' + str(5) + '\t' +  # width of header and width of each object.  
                    str(int(Data['height'])) + '\t' + str(Data['width']) + '\t' +  # (width, height)  
                    str(1) + '\t' +  # class  
                    str(x_min) + '\t' + str(y_min) + '\t' + str(x_max) + '\t' + str(y_max) + '\t' +  # xmin, ymin, xmax, ymax  
                    str(Data['img_name'])+'\n')  
            img_idx += 1
           

if __name__ == "__main__":

    convert(coco_filename = './data/hicfa-training/train_data/coco_training.json',
            lst_filename  = './data/hicfa-training/train_data/training.lst',
    )
    # python /home/greg/dev/3rdparty/mxnet/tools/im2rec.py --pack-label ./data/hicfa-training/test_data/test.lst ./data/hicfa-training/test_data
    # python /home/greg/dev/3rdparty/mxnet/tools/im2rec.py --pack-label ./data/hicfa-training/val_data/validation.lst ./data/hicfa-training/val_data
    # python /home/greg/dev/3rdparty/mxnet/tools/im2rec.py --pack-label ./data/hicfa-training/train_data/training.lst ./data/hicfa-training/train_data
    