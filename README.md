# DeepTemporalSeg

This repository contains code to learn a model for semantic segmentation of a 3D LiDAR scan 
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
* tfrecords files for the dataset generated from the KITTI tracking sequence by us
### 2.3. Training the model
All the files required for training and testing the model is in python_scripts folder. To train the model following script has to be executed.

```
python train_seg.py 

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
To test the model we provide the code for calculating the FPR-95 error. The model is tested on 50,000 positive and negative image patches from the testing data. This script prints the FPR-95 error, plots the curve between TPR and FPR, and stores the data used for plotting the curve.

```
python test_model.py

Parameters
--path_to_saved_model
--path_to_testing_data

```

#### 2.4.1. Example command for testing a trained model
```
python test_model.py --path_to_saved_model learned_models/my_model_retrain_55031  --path_to_testing_data ../dataset/testing_data.hdf5

```


## 3. C++ PCL Interface

### 3.1. Prerequisites

* Tensorflow
* [PCL 1.8] (https://github.com/PointCloudLibrary/pcl)
* [OpenCV] (https://github.com/opencv/opencv)
* [Thrift] (https://thrift.apache.org/download)

Thrift is required for both C++ and Python.

### 3.2. Installing

In the project directory

```
mkdir build
cd build
cmake ..
make
```

In case PCL 1.8 is not found, use -DPCL_DIR variable to specify the path of PCL installation.
```
cmake .. -DPCL_DIR:STRING=PATH_TO_PCLConfig.cmake
```

### 3.3. Downloading the test pointcloud

```
./download_test_pcd.sh
```

This will download the test pointcloud files used in the alignment experiment in the paper. The name format for the files is seq_scan_trackID_object.pcd. 'seq' corresponds to the sequence number from KITTI tracking benchmark. 'scan' is the scan used from the given sequence. 'trackID' is the object ID provided by the benchmark. For instance, '0011_126_14_object.pcd' and '0011_127_14_object.pcd' is the same object in two consecutive scans.

### 3.4. Downloading the models

```
./download_models.sh
```

This will download the trained model files. We provide the model for a feature descriptor learned simulataneously with a metric for matching as well as a feature descriptor learned using hinge loss. 'deep_3d_descriptor_matching' is the model for the descriptor using the learned metric, 'deep_3d_descriptor_hinge_loss' for the descriptor trained using hinge loss.

### 3.5. Using the learned descriptor with PCL

We provide a service and client API for using the learned feature descriptor with PCL.

All the Thrift related code and the python service file is in the folder python_cpp.

The service has to be started within the tensorflow environment.
```
python python_server.py

Parameters
--model_name
--using_hinge_loss

```

We provide two test files, the first one for computing a feature descriptor and the second one for matching the descriptors.

For computing feature descriptor

```
./compute_deep_3d_feature

Parameters
--path_to_pcd_file
--feature_neighborhood_radius (default: 1.6)
--sampling_radius (default: 0.4)

```

For visualizing the correspondences and using them to align the pointclouds (--use_ransac for inlier correspondences only)

```
./visualize_deep_3d_feature_correspondences

Parameters
--path_to_source_pcd_file
--sampling_radius_source
--path_to_target_pcd_file
--sampling_radius_target
--feature_neighborhood_radius
--use_learned_metric
--use_ransac

```

#### 3.5.1. Examples for visualizing the correspondences and the aligned pointcloud 

##### 3.5.1.1. Estimate the correspondences using the learned metric

```
python python_server.py --model_name ../models/deep_3d_descriptor_matching --use_hinge_loss 0

```

```
./visualize_deep_3d_descriptor_correspondences --path_to_source_pcd_file ../test_pcd/0011_1_2_object.pcd --sampling_radius_source 0.2 --path_to_target_pcd_file ../test_pcd/0011_2_2_object.pcd --sampling_radius_target 0.1 --feature_neighborhood_radius 1.6 --use_learned_metric 1 --use_ransac 0

```
Matched Keypoints             |Aligned Scans
:-------------------------:|:-------------------------:
![](http://deep3d-descriptor.informatik.uni-freiburg.de/corr_metric.png)  |  ![](http://deep3d-descriptor.informatik.uni-freiburg.de/aligned_metric.png)

##### 3.5.1.2. Estimate the correspondences using Euclidean distance

```
python python_server.py --model_name ../models/deep_3d_descriptor_hinge_loss --use_hinge_loss 1

```

```
./visualize_deep_3d_descriptor_correspondences --path_to_source_pcd_file ../test_pcd/0011_1_2_object.pcd --sampling_radius_source 0.2 --path_to_target_pcd_file ../test_pcd/0011_2_2_object.pcd --sampling_radius_target 0.1 --feature_neighborhood_radius 1.6 --use_learned_metric 0 --use_ransac 0

```
Matched Keypoints             |Aligned Scans
:-------------------------:|:-------------------------:
![](http://deep3d-descriptor.informatik.uni-freiburg.de/corr_hinge.png)  |  ![](http://deep3d-descriptor.informatik.uni-freiburg.de/aligned_hinge.png)


