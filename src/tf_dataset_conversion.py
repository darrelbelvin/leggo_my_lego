import tensorflow as tf

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
import pandas as pd
import numpy as np
import pickle
import hashlib


# flags = tf.app.flags
# flags.DEFINE_string('source_path', '', 'Path to source images with metadata')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('num_shards', '10', 'number of shards')
# FLAGS = flags.FLAGS


def create_tf_example(example, source_path):
  # TODO(user): Populate the following variables from your example.
  height = 416 # Image height
  width = 416 # Image width
  filename = example[0] # Filename of the image. Empty if image is not from file
  image_format = 'png' # b'jpeg' or b'png'

  with tf.io.gfile.GFile(source_path + filename, 'rb') as fid:
    encoded_png = fid.read()

  key = hashlib.sha256(encoded_png).hexdigest()
  # xmin, ymin, xmax, ymax, i
  xmins = example[1::5] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example[3::5] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = example[2::5] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example[4::5] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  
  class_names = ['LEGS', 'TORSO', 'HEAD', 'HATS', 'MINIFIG', '']
  class_names = [s.encode('utf8') for s in class_names]

  classes = example[5::5] # List of string class name of bounding box (1 per box)
  classes_text = [class_names[i] for i in classes] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


# def main(_):
#   metadata = pickle.load(FLAGS.source_path + 'metadata.pkl', 'rb')
#   examples = enumerate(list(metadata['bboxes']))

#   output_filebase=FLAGS.output_path + '.record'
#   num_shards = int(FLAGS.num_shards)

#   with contextlib2.ExitStack() as tf_record_close_stack:
#     output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
#         tf_record_close_stack, output_filebase, num_shards)
#     for index, example in examples:
#       tf_example = create_tf_example(example)
#       output_shard_index = index % num_shards
#       output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

  # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # for example in examples:
  #   tf_example = create_tf_example(example)
  #   writer.write(tf_example.SerializeToString())

  # writer.close()


if __name__ == '__main__':
  #tf.app.run()

  #import sys
  #sys.path.append('/home/darrel/Documents/gal/projects/leggo_my_legs/google_cloud/tf_models/research/')
  source_path = '/home/darrel/Documents/gal/projects/leggo_my_legs/datagen/minifig_rendered_try2/images/combined/'
  metadata = pickle.load(open(source_path + 'metadata.pkl', 'rb'))
  examples = enumerate(list(metadata['bboxes']))

  output_filebase= '/home/darrel/Documents/gal/projects/leggo_my_legs/datasets/minifigs_3000_v1/lego_shards.record'
  num_shards = 16

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
    for index, example in examples:
      tf_example = create_tf_example(example, source_path)
      output_shard_index = index % num_shards
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
