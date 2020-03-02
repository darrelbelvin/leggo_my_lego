import tensorflow as tf

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
import pandas as pd
import pickle


flags = tf.app.flags
flags.DEFINE_string('source_path', '', 'Path to source images with metadata')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('num_shards', '10', 'number of shards')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = example[0] # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = 'png' # b'jpeg' or b'png'

  xmins = list(np.array(example[1::6])/width) # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = list(np.array(example[2::6])/width) # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = list(np.array(example[3::6])/height) # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = list(np.array(example[4::6])/height) # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = example[5::5] # List of string class name of bounding box (1 per box)
  classes = [6::6] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  metadata = pickle.load(FLAGS.source_path + 'metadata.pkl')
  examples = metadata['bboxes'].apply(create_tf_example)

  # TODO(user): Write code to read in your dataset to examples variable

  output_filebase=FLAGS.output_path + '.record'
  num_shards = int(FLAGS.num_shards)

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
    for index, example in examples:
      tf_example = create_tf_example(example)
      output_shard_index = index % num_shards
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

  # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # for example in examples:
  #   tf_example = create_tf_example(example)
  #   writer.write(tf_example.SerializeToString())

  # writer.close()


if __name__ == '__main__':
  tf.app.run()