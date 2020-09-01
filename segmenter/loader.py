from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
from mxnet import image, nd
import numpy as np
import os
import mxnet as mx

# https://mxnet.apache.org/versions/1.2.1/tutorials/gluon/datasets.html

def read_images(root):
    img_dir = os.path.join(root, 'image')
    mask_dir = os.path.join(root, 'mask')
    mask_filenames = os.listdir(mask_dir)
    img_filenames = os.listdir(img_dir) 

    if len(img_filenames) != len(mask_filenames):
        raise Exception('Wrong size')

    features, labels = [None] * len(img_filenames), [None] * len(mask_filenames)
    for i, fname in enumerate(img_filenames):
        features[i] = image.imread(os.path.join(img_dir, fname))
        labels[i] = image.imread(os.path.join(mask_dir, fname))

    return features, labels

class SegDataset(gdata.Dataset):
    def __init__(self, root, transform = None, colormap = None, classes=None):
       features, labels = read_images(root)
       self.rgb_mean = nd.array([0.448, 0.456, 0.406])
       self.rgb_std = nd.array([0.229, 0.224, 0.225])
       self.transform = transform
       self.features=[self.normalize_image(feature) for feature in features]
       self.labels=labels

       self.colormap = colormap
       self.classes = classes
       self.colormap2label = None

    def normalize_image(self, img):
        return (img.astype('float32') / 255.0 - self.rgb_mean) / self.rgb_std


    def label_indices(self,img):  
        if self.colormap2label is None:
            self.colormap2label = nd.zeros(256**3)

            for i, cm in enumerate(self.colormap):
                self.colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        data = img.astype('int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return self.colormap2label[idx]

    def __getitem__(self, idx):
        feature, label = self.features[idx], self.labels[idx]
        if self.transform is not None:
            return self.transform(feature, label)
        # 2x512x512x3 
        # convert into BxCxHxW        
        # print(feature.shape)
        # print(feature.transpose((2, 0, 1)).shape)
        # return feature, label
        return feature.transpose((2, 0, 1)), self.label_indices(label)
        
    def __len__(self):
        return len(self.features)

def _get_batch(batch, ctx, is_even_split=True):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return gutils.split_and_load(features, ctx, even_split=is_even_split), gutils.split_and_load(labels, ctx, even_split=is_even_split), features.shape[0]

if __name__ == '__main__':
    print('Loader test')
    # /data/train
    # image , mask
    # the RGB label of images and the names of lables
    COLORMAP = [[0, 0, 0], [255, 255, 255]]
    CLASSES = ['background', 'form']

    dataset = SegDataset('./data/train', transform = None, colormap=COLORMAP, classes=CLASSES)
    loader = mx.gluon.data.DataLoader(dataset, 1, num_workers=1)
    ctx = [mx.cpu()]
    for i, batch in enumerate(loader):
        features, labels = batch
        feature = gutils.split_and_load(features, ctx, even_split=True)
        label = gutils.split_and_load(labels, ctx, even_split=True)
        print(feature[0].shape)
        print(labels[0].shape)
        # print(labels.shape)
