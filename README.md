# cmake-mxnet-template

cmake project template that includes mxnet, leptonica, opencv preconfigured

## CMAKE Configuration

Edit `CMakeLists.txt` in root directory of the project and change `MXNET_ROOT` parameter to point to your installation directory.

```bash
SET(ENV{MXNET_ROOT} "/home/gbugaj/dev/3rdparty/mxnet")
```

Project has following dependencies

* MxNet - FindMxNet.cmake
* Leptonica - FindLeptonica.cmake
* OpenCV - FindOpenCV.cmake

## Building


## Creating 

# Minist as JPG

https://www.kaggle.com/scolianni/mnistasjpg
https://stats.stackexchange.com/questions/166600/mnist-dataset-black-or-white-background
https://www.kaggle.com/scolianni/mnistasjpg

## Resources

https://github.com/zalandoresearch/fashion-mnist
https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=95651665
https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html
https://stackoverflow.com/questions/39293922/convert-between-opencv-mat-and-leptonica-pix#39293974
https://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html


Multi-channel images 
    https://stackoverflow.com/questions/55297514/mxnet-c-inference-with-mxpredsetinput-segmentation-fault
computing mean image
    https://github.com/apache/incubator-mxnet/issues/371

# Python testing notes
~/dev/3rdparty/mxnet/example/image-classification$ python3 ./train_mnist.py  --network lenet  --model-prefix /home/greg/dev/rms/matchbox/data/models/py-mlp/lenet

# Network visualization 
https://lutzroeder.github.io/netron/

## Tensorboard / mxboard
https://medium.com/apache-mxnet/mxboard-mxnet-data-visualization-2eed6ae31d2c
https://gist.github.com/skirdey/8b9bd6cc20e958bc57e76004410f17ff
