"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'man':
        return 1
    elif row_label == 'person':
        return 2
    elif row_label == 'window':
        return 3
    elif row_label == 'tree':
        return 4
    elif row_label == 'building':
        return 5
    elif row_label == 'shirt':
        return 6
    elif row_label == 'wall':
        return 7
    elif row_label == 'woman':
        return 8
    elif row_label == 'sign':
        return 9
    elif row_label == 'sky':
        return 10
    elif row_label == 'ground':
        return 11
    elif row_label == 'grass':
        return 12
    elif row_label == 'table':
        return 13
    elif row_label == 'pole':
        return 14
    elif row_label == 'head':
        return 15
    elif row_label == 'light':
        return 16
    elif row_label == 'water':
        return 17
    elif row_label == 'car':
        return 18
    elif row_label == 'hand':
        return 19
    elif row_label == 'hair':
        return 20
    elif row_label == 'people':
        return 21
    elif row_label == 'leg':
        return 22
    elif row_label == 'trees':
        return 23
    elif row_label == 'clouds':
        return 24
    elif row_label == 'ear':
        return 25
    elif row_label == 'plate':
        return 26
    elif row_label == 'leaves':
        return 27
    elif row_label == 'door':
        return 28
    elif row_label == 'fence':
        return 29
    elif row_label == 'pants':
        return 30
    elif row_label == 'eye':
        return 31
    elif row_label == 'train':
        return 32
    elif row_label == 'floor':
        return 33
    elif row_label == 'chair':
        return 34
    elif row_label == 'road':
        return 35
    elif row_label == 'hat':
        return 36
    elif row_label == 'street':
        return 37
    elif row_label == 'snow':
        return 38
    elif row_label == 'wheel':
        return 39
    elif row_label == 'shadow':
        return 40
    elif row_label == 'jacket':
        return 41
    elif row_label == 'nose':
        return 42
    elif row_label == 'boy':
        return 43
    elif row_label == 'line':
        return 44
    elif row_label == 'shoe':
        return 45
    elif row_label == 'clock':
        return 46
    elif row_label == 'sidewalk':
        return 47
    elif row_label == 'tail':
        return 48
    elif row_label == 'boat':
        return 49
    elif row_label == 'cloud':
        return 50
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
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
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.compat.v1.app.run()
