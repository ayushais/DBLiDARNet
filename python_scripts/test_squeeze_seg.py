from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import cv2
import tensorflow as tf
def main():
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('model_name', '',
                             """model name""")
  tf.app.flags.DEFINE_string('validation_record_filename', '',
                             """path to test record""")
  tf.app.flags.DEFINE_string('is_visualize', 'no',
                             """is_visualize""")

  tf.app.flags.DEFINE_integer('image_width', 512, """image width""")
  if FLAGS.model_name == '':
    print('model name not specified')
    return
  car = [165, 194, 102]
  pedestrain = [98, 141, 252]
  bicyclist = [203, 160, 141]
  with tf.Session() as sess:
    model_name = FLAGS.model_name + '.ckpt.meta'
    saver = tf.train.import_meta_graph(model_name)
    model_name = FLAGS.model_name + '.ckpt'
    saver.restore(sess, model_name)
    graph = tf.get_default_graph()
    input_data = graph.get_tensor_by_name("input_data:0")
    # labels = graph.get_tensor_by_name("labels:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    prediction = graph.get_tensor_by_name("prediction/conv:0")
    print('loaded graph')
    record_iterator = tf.python_io.tf_record_iterator(FLAGS.validation_record_filename)
    intersection_total = np.zeros((1, 3), np.float32)
    union_total = np.zeros((1, 3), np.float32)

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
      gt_mask = gt_mask.reshape((height, width))
      feed_dict = {input_data: image, keep_prob: 1, is_training: False}
      predicted_mask = sess.run(prediction, feed_dict=feed_dict)
      predicted_mask = np.squeeze(predicted_mask, axis=0)
      predicted_mask = np.argmax(predicted_mask, axis=2)
      if FLAGS.is_visualize == 'yes':
        label_color = np.ones((64, FLAGS.image_width, 3), np.uint8)
        label_color *= 255
        prediction_color = np.ones((64, FLAGS.image_width, 3), np.uint8)
        prediction_color *= 255
      for class_id in range(1, 4):
        gt_mask_class = gt_mask == class_id
        predicted_mask_class = (predicted_mask == class_id)
        intersection = np.sum(np.logical_and(gt_mask_class,
                                             predicted_mask_class))
        union = np.sum(predicted_mask_class) + np.sum(gt_mask_class) - \
        intersection
        intersection_total[0, class_id-1] += intersection
        union_total[0, class_id-1] += union
        if FLAGS.is_visualize == 'yes':
          if class_id == 1:
            label_color[gt_mask_class, :] = car
            prediction_color[predicted_mask_class, :] = car
          if class_id == 2:
            label_color[gt_mask_class, :] = pedestrain
            prediction_color[predicted_mask_class, :] = pedestrain
          if class_id == 3:
            label_color[gt_mask_class, :] = bicyclist
            prediction_color[predicted_mask_class, :] = bicyclist
          viz_combined = np.zeros((128, FLAGS.image_width, 3), np.uint8)
          viz_combined[0:64, :, :] = label_color
          viz_combined[64:128, :, :] = prediction_color

      if FLAGS.is_visualize == 'yes':
        cv2.namedWindow("visualization")
        cv2.imshow("visualization", viz_combined)
        cv2.waitKey(30)
    IoU = intersection_total/union_total
    mean_IoU = np.mean(IoU)
    print(mean_IoU)
    print(IoU)
if __name__ == '__main__':
  main()


