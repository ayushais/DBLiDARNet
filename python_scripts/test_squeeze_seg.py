from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from sys import argv
import h5py
import random
import numpy as np
import cv2
import tensorflow as tf
import time
FLAGS = None


def main():
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('model_name', '', 
      """model name""")
  tf.app.flags.DEFINE_string('validation_record_filename', '',
      """path to test record""")
  
  if FLAGS.model_name == '':
    print ('model name not specified')
    return

  with tf.Session() as sess:

    image_width = 512
    image_height = 64
    batch_size = 1
    num_classes = 4
    num_channels = 1
    model_name = FLAGS.model_name + '.ckpt.meta'
    saver = tf.train.import_meta_graph(model_name)
    model_name = FLAGS.model_name + '.ckpt'
    saver.restore(sess,model_name)
    graph = tf.get_default_graph()
    input_data = graph.get_tensor_by_name("input_data:0")
    # labels = graph.get_tensor_by_name("labels:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    prediction = graph.get_tensor_by_name("prediction/conv:0")
    print('loaded graph')
    record_iterator = tf.python_io.tf_record_iterator(FLAGS.validation_record_filename)
    intersection_total = np.zeros((1,num_classes-1),np.float32)
    union_total = np.zeros((1,num_classes-1),np.float32)

    for string_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(string_record)
      
      height = int(example.features.feature['height'].int64_list.value[0])
      width = int(example.features.feature['width'].int64_list.value[0])
      image_string = example.features.feature['image_raw'].bytes_list.value[0]
      mask_string = example.features.feature['mask_raw'].bytes_list.value[0]
      
      image = np.fromstring(image_string, dtype=np.float32)
      image = image.reshape((1, height, width, 5)) 
      gt_mask = np.fromstring(mask_string, dtype=np.float32)
      gt_mask = gt_mask.reshape((height, width, 1))
      feed_dict = {input_data:image, keep_prob:1, is_training:False}
      start_time = time.time()
      predicted_mask = sess.run(prediction, feed_dict=feed_dict)
      predicted_mask = np.squeeze(predicted_mask, axis=0)
      predicted_mask = np.argmax(predicted_mask, axis=2)
      for class_id in range(1,4):
        gt_mask_class = gt_mask == class_id
        predicted_mask_class = (predicted_mask == class_id)
     #   gt_label = np.squeeze(gt_label,axis=2)
        combined = np.zeros((128, 512),np.float32)
        combined[0:64,:] = gt_mask_class
        combined[64:128,:] = predicted_mask_class
        cv2.namedWindow("prediction")
        cv2.imshow("prediction",np.float32(combined))
        intersection = np.sum(np.logical_and(gt_mask_class, 
                                predicted_mask_class))
        union = np.sum(predicted_mask_class) + np.sum(gt_mask_class) - intersection
        print(intersection/union)

        cv2.waitKey()
        intersection_total[0,classes-1]+=intersection
        union_total[0,classes-1]+=union

      # predicted_mask = np.argmax(predicted_mask, axis=2)
      

#       predicted_mask = predicted_mask > 0
      # cv2.namedWindow("mask")
      # cv2.imshow("mask", np.float32(predicted_mask))
      # cv2.waitKey()

    
    IoU = intersection_total/union_total
    mean_IoU = np.mean(IoU)
    print(mean_IoU)
    print(IoU)












    # f = open('test.txt','r')
    # for line in f:
      # input_file = line.replace('\n','')
# #       line = line.split()
      # # input_dir = line[0]
      # # number_files = int(line[1])
    # # f.close()

    # hdf5_file = h5py.File(input_file,'r') 
    # a_group_key_data = hdf5_file.keys()[0]
    # data_d_test = list(hdf5_file[a_group_key_data])
    # test_data = np.array(data_d_test)

    # test_data = test_data[:,:,:,4]
    # test_data = np.expand_dims(test_data, axis=3)
    # #test_data = np.transpose(test_data,(0,2,3,1))
    # a_group_key_label = hdf5_file.keys()[1]
    # data_l_test = list(hdf5_file[a_group_key_label])
    # test_label_data = np.array(data_l_test)
    # #test_label_data = np.transpose(data_l_test,(0,2,3,1))
    # hdf5_file.close()
    # print('loaded data')
    # data_size = test_data.shape[0]
    # #train_data = graph.get_tensor_by_name("train_data:0")
    # #train_labels = graph.get_tensor_by_name("train_labels:0")
 
    # train_data = graph.get_tensor_by_name("input_data:0")
    # train_labels = graph.get_tensor_by_name("labels:0")
    
    # keep_prob = graph.get_tensor_by_name("keep_prob:0")
    # is_training = graph.get_tensor_by_name("is_training:0")
    # #crop = graph.get_tensor_by_name("crop_upconv_5/conv:0")##only for our architec 8s
    # crop = graph.get_tensor_by_name("prediction/conv:0")##only for our architec 8s
    # #crop = graph.get_tensor_by_name("control_dependency_3:0")##only for fastnet 8s
    # #crop = graph.get_tensor_by_name("control_dependency_7:0")##only for fastnet 4s
    # #crop = graph.get_tensor_by_name("control_dependency_11:0")##only for fastnet 2s
    # #crop = graph.get_tensor_by_name("Final_decoder/BiasAdd:0")##only for inverted residual
    # IoU  = 0
    
    # intersection_total = np.zeros((1,num_classes-1),np.float32)
    # union_total = np.zeros((1,num_classes-1),np.float32)

    # #intersection_total = np.zeros((1,1),np.float32)
    # #union_total = np.zeros((1,1),np.float32)
    # total_time = 0.0
    # for index in range(0,data_size):
      # #if(index > 10):
      # #  break;
      # data_training = np.zeros([batch_size,image_height,image_width,num_channels])
      # data_training[0,:,:,:] = test_data[index,:,:,:]
      # data_training = np.reshape(data_training,(batch_size,image_height,image_width,num_channels))
      # label_training_one_channel = test_label_data[index,:,:]
# #           label_training = np.zeros([batch_size,image_height,image_width,num_classes])
          # # for row in range(0,64):
            # # for column in range(0,324):
              # # if(int(label_training_one_channel[row,column,0]) > 0):
                # # label_training[0,row,column,int(label_training_one_channel[row,column,0])] = 1
              # # else:
                # # label_training[0,row,column,int(label_training_one_channel[row,column,0])] = 1
          
      # feed_dict = {train_data:data_training,keep_prob:1,is_training:False}
      # start_time = time.time()
      # predictions = sess.run([crop],feed_dict=feed_dict)

      # end_time = time.time()
      # total_time += float(end_time - start_time)
      # predictions = np.array(predictions)
      # predictions = predictions[:,0,:,:,:]
      # predictions = np.squeeze(predictions,axis=0)
      # predictions = np.argmax(predictions,axis=2)
      

      
      # #combined = np.zeros((128,324),np.float32)

      # #label_training = np.squeeze(label_training_one_channel,axis=2)
      # #combined[0:64,:] = np.array(label_training)
      # #combined[64:128,:] = np.array(predictions)
     # # combined[combined == 1] = 1.0
     # # combined[combined == 2] = 0.74
     # # combined[combined == 3] = 0.5

      # #cv2.namedWindow("prediction")
      # #cv2.imshow("prediction",np.float32(predictions))
      # #cv2.waitKey()
      # #cv2.imwrite('/home/dewan/kitti_tracking_results/squeeze_seg_depth_separable_decoder_shuffle_again_195/' + str(index) + '_.png', np.int8(predictions))
    # #  label_training_one_channel = label_training_one_channel > 0
      # for classes in range(1,num_classes):
        # gt_label = (label_training_one_channel == classes)
        # predictions_class = (predictions == classes)
     # #   gt_label = np.squeeze(gt_label,axis=2)
# #        combined = np.zeros((128,324),np.float32)
# #        combined[0:64,:] = gt_label
# #        combined[64:128,:] = predictions_class
# #        cv2.namedWindow("prediction")
# #        cv2.imshow("prediction",np.float32(combined))
# #        cv2.waitKey()
          
     # #   print('class %d' % classes)

        # intersection = np.sum(np.multiply(predictions_class,gt_label))
        # union = intersection + (np.sum(predictions_class) - intersection) + (np.sum(gt_label) - intersection)
        # intersection_total[0,classes-1]+=intersection
        # union_total[0,classes-1]+=union
# #        if(union > 0):
 # #         print('IoU %f, %d,%d,%d,%d' % ((intersection/union),intersection,union,index,classes))
        # #print(intersection_total/union_total)
    # print(intersection_total/union_total)
    # #raw_input()
    # filename_save = '../results/' + filename + '_' + save_iter +'.csv'
  
    # f = open(filename_save,"w")
    # IoU = intersection_total/union_total
    # mean_IoU = np.mean(IoU)
    # print(mean_IoU)
    # f.write(str(IoU))
    # f.write(str(mean_IoU))

    

    # f.close()
    # print(total_time/data_size)
    
     # IoU+=(intersection/union)
#      print('IoU: %f' % (intersection/union))
      



#           predictions = np.array(predictions[0,:,:,:])
          # predictions = np.reshape(predictions,(image_height,image_width,num_classes))

          # predictions_cropped = np.argmax(predictions,axis=2)
          # predictions_cropped = predictions_cropped > 0

          # predictions = np.concatenate((left_side_complete,predictions,right_side_complete),axis=1)

          # volume = np.concatenate((left_side,volume,right_side),axis=1)
          # center = np.concatenate((left_side,center,right_side),axis=1)
          # rotation = np.concatenate((left_side_rotation,rotation,right_side_rotation),axis=1)

          # volume = np.float32(volume)
          # center = np.float32(center)
          # rotation = np.float32(rotation)
 


          # image_file = results_dir +  filename + '/' + str(number_files) + '/volume_' + str(file_number) + '.exr'
          # cv2.imwrite(image_file,volume)
          
          
          # image_file = results_dir +  filename + '/' + str(number_files) + '/center_' + str(file_number) + '.exr'
          # cv2.imwrite(image_file,center)
          
          # image_file = results_dir +  filename + '/' + str(number_files) + '/rotation_' + str(file_number) + '.exr'
          # cv2.imwrite(image_file,rotation)


     


        

          # out = np.argmax(predictions,axis=2)
          # image_file = results_dir +  filename + '/' + str(number_files) + '/' +  str(file_number) + '.tiff'
          # img = Image.fromarray(np.uint8(out))

          # out_binary = out > 0
          # img.save(image_file)
          
          # image = predictions[:,:,0]

          # image_file = results_dir +  filename + '/' + str(number_files) +'/bg_' + str(file_number) + '.tiff'
          # img = Image.fromarray(image)
          # img.save(image_file)


          # # for class_id in range(1,num_classes):

              # # image = predictions[:,:,class_id]
              # # image_file = results_dir +  filename + '/' + str(number_files) +'/fg' + str(class_id) + '_' + str(file_number) + '.tiff'
              # # img = Image.fromarray(image)
              # # img.save(image_file)
         
          # label_mask = (label_training_one_channel > 0)
          # # label_volume = np.reshape(label_volume,(-1,3))
          # # label_center = np.reshape(label_center,(-1,3))
          # label_mask = np.reshape(label_mask,(-1,1))
          # # label_rotation = np.reshape(label_rotation,(-1,1))
          # predictions_cropped = np.reshape(predictions_cropped,(-1,1))
          # out_binary = np.reshape(out_binary,(-1,1))


#           print(label_mask.shape)
          # print(volume.shape)
          # print(label_center.shape)

          # center_loss = np.multiply((label_center - center_cropped),predictions_cropped)
          # center_loss_val = np.sum(np.power(center_loss,2))
          
          # volume_loss = np.multiply((label_volume - volume_cropped),predictions_cropped)
          # volume_loss_val = np.sum(np.power(volume_loss,2))
          
          # rotation = np.reshape(rotation,(-1,1))
          # rotation_loss = np.multiply((label_rotation - rotation),out_binary)

          # rotation_loss_val = np.sum(np.power(rotation_loss,2))

          # if(loss_vec.shape[0] == 0):
            # loss_vec = np.append(loss_vec,[center_loss_val/(image_width * image_height),volume_loss_val/(image_width * image_height),rotation_loss_val/(image_width * image_height)],axis=0)
            # loss_vec = np.reshape(loss_vec,(1,3))
          # else:
            # loss_combine = [center_loss_val/(image_width * image_height),volume_loss_val/(image_width * image_height),rotation_loss_val/(image_width * image_height)]
            # loss_combine = np.reshape(loss_combine,(1,3))
            # loss_vec = np.append(loss_vec,loss_combine,axis=0)

#           print(loss_vec)
          # raw_input()
            
            




          # print(rotation_loss_val/(image_width * image_height))

          
          # center_loss_total += (center_loss_val/(image_width * image_height))
          # volume_loss_total += (volume_loss_val/(image_width * image_height))
          # rotation_loss_total += (rotation_loss_val/(image_width * image_height))
    # np.savetxt('train_loss.txt',loss_vec,fmt='%10.5f',delimiter=',')
#     print(center_loss_total/float(number_files))
    # print(volume_loss_total/float(number_files))
    # print(rotation_loss_total/float(number_files))

                       

      

if __name__ == '__main__':
  main()


