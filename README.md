# DeepTemporalSeg

This repository contains code to learn a model for semantic segmentation of 3D LiDAR scans 
#### Related Publication

Ayush Dewan, Wolfram Burgard  
**[DeepTemporalSeg: Temporally Consistent Semantic Segmentation of 3D
LiDAR Scans](http://ais.informatik.uni-freiburg.de/publications/papers/dewan18iros.pdf)**  
*IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, Spain, 2018*  

## 1. License

This software is released under GPLv3. If you use it in academic work, please cite:

```
@inproceedings{dewan2018iros,
  author = {Ayush Dewan and Wolfram Burgard},
  title = {DeepTemporalSeg: Temporally Consistent Semantic Segmentation of 3D
LiDAR Scans 
},
  booktitle = {Proc.~of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  address = {},
  year = {2019},
  url = {http://ais.informatik.uni-freiburg.de/publications/papers/dewan18iros.pdf}
}
```


## 2. Training the Network 

### 2.1. Prerequisites

* Tensorflow
* Pyhton 3.6
* H5py

### 2.2. Dataset

```
./download_dataset.sh

```

This will download the following datasets:
* tfrecord files for the dataset from https://github.com/BichenWuUCB/SqueezeSeg
* tfrecords files for our dataset generated from the KITTI tracking benchmark
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

```


#### 2.3.1. Example commands for starting the training 

```
python train_seg.py --model_name lidar_segmentation --train_record_filename ../datasets/squeeze_seg/squeeze_seg_training/ --validation_record_filename ../datasrts/squeeze_seg/squeeze_seg_val/ --image_width 512 --batch_size 2
```

### 2.4. Testing the model

```
./download_models.sh

```
This will download the models trained on the dataset from https://github.com/BichenWuUCB/SqueezeSeg
and KITTI tracking bencmark




```
test.py 

Parameters

--model_name 
--validation_record_filename
--is_visualize (default: no)
--image_width (default: 512)


```

#### 2.4.1. Example command for testing a trained model
```
python test.py --model_name ../models/squeeze_seg --validation_record_filename ../datasets/squeeze_seg_validation.records --is_visualize yes

```






