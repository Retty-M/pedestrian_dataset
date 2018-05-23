# coding=utf-8
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
import json

import numpy as np
import tensorflow as tf

from PIL import Image
from collections import namedtuple
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('json_input', '', 'Path to the Json input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def split(df):
    grouped = []
    data = namedtuple('data', ['filename', 'object'])
    for set in df:
        if int(set[3:]) < 6:
            for video in df[set]:
                for frame in df[set][video]['frames']:
                    filename = set + '_' + video + '_' + frame + '.png'
                    grouped.append(data(filename, df[set][video]['frames'][frame]))
    # print(grouped)
    return grouped


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = 'png'.encode('utf8')
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    classes_text = []

    for row in group.object:
        if row['lbl'] == 'person':
            pos = np.int0(np.around(row['pos']))
            xmin = pos[0]
            ymin = pos[1]
            xmax = pos[0] + pos[2]
            ymax = pos[1] + pos[3]
            xmins.append(xmin / width)
            xmaxs.append(xmax / width)
            ymins.append(ymin / height)
            ymaxs.append(ymax / height)
            print(xmins[-1], xmaxs[-1], ymins[-1], ymaxs[-1])
            classes_text.append('person')
            classes.append(1)

    if len(xmins) <= 0:
        return
    print(len(xmins))

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
        # 'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = '/home/ubuntu/Datasets/caltech/data/images'
    with open(FLAGS.json_input, 'r') as f:
        examples = json.load(f)
        grouped = split(examples)
        for group in grouped:
            tf_example = create_tf_example(group, path)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
