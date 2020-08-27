# Document-SSD-MXNet
SSD Detector trained on custom document dataset

# Installation

## 1. Prepare Dataset
### 1.1 Prepare training images

* Create `train` and `val` datasets  
* Move all image file to `data/raw` folder to easily manipulate

	`mkdir -p ./data/raw`


### 1.2 Create record data
Follow `lst-rec-prep.ipynb` to create List and RecordIO file - they are the dataset format developed by MXNet. To support the training procedure, it is preferred to utilize data in binary format rather than raw images as well as the annotations.

If you are not familiar with this process, refer this tutorial [[link]](https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html)
#### 1.2.1 Create LST file
Follow step 0 to step 2 to first create `.lst` file. By performing these steps, 2 file `train.lst` and `val.lst` are generated.

#### 1.2.2 Create REC file
After obtaining the `.lst` files. Start generating `.rec` by using built-in feature from MXNet

`python /path/to/incubator-mxnet/tools/im2rec.py train.lst data/raw/ --pass-through --pack-label`

`python /path/to/incubator-mxnet/tools/im2rec.py val.lst data/raw/ --pass-through --pack-label`

It's gonna take a few seconds to create record files for `train` and `val` datasets. After finishing this step, 4 files are created:
* train.idx
* train.rec
* val.idx
* val.rec

Move all generated files to `datasets` directory

`mkdir datasets`

`mv *.lst *.idx *.rec datasets`


## cleanup

```
ps aux | grep 'train_ssd' | awk '{print $2}' | xargs kill
```

## Notes 

https://d2l.ai/chapter_computer-vision/object-detection-dataset.html

https://pastebin.com/msqw0MQh
https://discuss.mxnet.io/t/is-this-a-correct-way-to-prepare-custom-data-for-yolo-v3-detector/2691
https://github.com/diewland/pika-dataset-v3

## Contains full dataset
https://discuss.mxnet.io/t/using-faster-rcnn-on-other-things-than-voc-coco/2978


## CVAT
http://localhost:8080/tasks?page=1
http://localhost:8080/admin/



https://www.researchgate.net/publication/320243569_Table_Detection_Using_Deep_Learning/download
2017_Deep_Table_ICDAR.pdf