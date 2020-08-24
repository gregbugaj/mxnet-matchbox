"""01. Predict with pre-trained SSD models
==========================================
"""

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

######################################################################
# Load a pretrained model
# -------------------------
net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

######################################################################
# Pre-process an image
#  Here we specify that we resize the short edge of the image to 512 px. But you can
# feed an arbitrarily sized image.

# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

######################################################################
# Inference and display
# ---------------------
#
# The forward function will return all detected bounding boxes, and the
# corresponding predicted class IDs and confidence scores. Their shapes are
# `(batch_size, num_bboxes, 1)`, `(batch_size, num_bboxes, 1)`, and
# `(batch_size, num_bboxes, 4)`, respectively.

class_IDs, scores, bounding_boxes = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],class_IDs[0], class_names=net.classes)
plt.show()