# DeepTemporalSeg

This repository contains code to learn a model for semantic segmentation of 3D LiDAR scans 



<img src="http://deep-temporal-seg.informatik.uni-freiburg.de/ezgif.com-video-to-gif_small.gif" width="580" height="394" align="center" />


## 1. License

This software is released under GPLv3. If you use it in academic work, please cite:

```
@article{dewan-deeptemporalseg,
  author = {Ayush Dewan and Wolfram Burgard},
  title = {DeepTemporalSeg: Temporally Consistent Semantic Segmentation of 3D LiDAR Scans},
  booktitle = {https://arxiv.org/abs/1906.06962},
  year = {2019},
  url = {http://deep-temporal-seg.informatik.uni-freiburg.de/dewan_deep_temporal_seg.pdf}
}
```


## 2. Training the Network 

### 2.1. Prerequisites

* Tensorflow 
* Pyhton 3.6

### 2.2. Dataset

```
./download_dataset.sh

```

This will download the following datasets:
* tfrecord files for the dataset from https://github.com/BichenWuUCB/SqueezeSeg
* tfrecords files for our dataset generated from the KITTI tracking benchmark. The details regarding the dataset are described in the paper. 
### 2.3. Training the model
All the files required for training and testing the model is in python_scripts folder. To train the model following script has to be executed.

```
train_seg.py 

Parameters
--model_name (default: lidar_segmentation)
--train_record_filename
--validation_record_filename
--log_dir
--path_to_store_models (default: learned_models/)
--learning_rate (default: 0.0001)
--eta (default: 0.0005)
--total_epochs (default: 200)
--batch_size (default: 2)
--image_height (default: 64)
--image_width (default: 324)
--num_channels (default: 5)
--num_classes (default: 4)
--growth (default: 16)
--gpu (default: 0)

```


#### 2.3.1. Example commands for starting the training 

```
python train_seg.py --model_name lidar_segmentation --train_record_filename ../datasets/squeeze_seg_train/ --validation_record_filename ../datasets/squeeze_seg_validation/ --image_width 512 --batch_size 2
```

### 2.4. Testing the model

```
./download_models.sh

```
This will download the models trained on the dataset from https://github.com/BichenWuUCB/SqueezeSeg
and KITTI tracking benchmark




```
test.py 

Parameters

--model_name 
--validation_record_filename
--is_visualize (default: no)
--image_width (default: 512)
--gpu (default: 0)

```

#### 2.4.1. Example command for testing a trained model
```
python test.py --model_name ../models/squeeze_seg --validation_record_filename ../datasets/squeeze_seg_validation/squeeze_seg_validation.records --is_visualize yes

```






