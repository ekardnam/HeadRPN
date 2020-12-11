import sys
sys.path.insert(0, '.')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import argparse
import random

from head_rpn.dataset import (
    load_kaggle_annotations,
    load_xml_annotations,
    aggregate_annotations
)
from math import ceil

TFRECORD_IMG_COUNT = 192

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates TFRecord files from a given Kaggle dataset')
    parser.add_argument('labels', type=str, help='Path to the labels.csv file')
    parser.add_argument('boxes', type=str, help='Path to the boxes.csv file')
    parser.add_argument('scut_annotations_folder', type=str, help='The folder containing SCUT annotations')
    parser.add_argument('kaggle_basepath', type=str, help='The base path for the Kaggle images')
    parser.add_argument('scut_basepath', type=str, help='The base path for SCUT images')
    parser.add_argument('--out', type=str, default='out/data-{}.tfrecord', help='Format string for output TFRecords')
    parser.add_argument('--img-count', type=int, default=TFRECORD_IMG_COUNT, help='The number of images in each TFRecord file')

    args = parser.parse_args()

    kaggle_annotations = load_kaggle_annotations(args.labels, args.boxes)
    scut_annotations = load_xml_annotations(args.scut_annotations_folder)

    annotations = aggregate_annotations(
                                        [kaggle_annotations, scut_annotations],
                                        [args.kaggle_basepath, args.scut_basepath]
                                       )
    random.shuffle(annotations)                                   

    annotation_count = len(annotations)

    for i in range(ceil(annotation_count / TFRECORD_IMG_COUNT)):
        filename = args.out.format(i)
        annotations_to_process = annotations[i * TFRECORD_IMG_COUNT : (i + 1) * TFRECORD_IMG_COUNT]
        with tf.python.python_io.TFRecordWriter(args.out.format(i)) as tfwriter:
            for annotation in annotations_to_process:
                image = tf.io.read_file(annotation["image_filename"])
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(image.numpy()),
                    'image_width': _int64_feature(annotation['image_width']),
                    'image_height': _int64_feature(annotation['image_height']),
                    'object_count': _int64_feature(annotation['object_count']),
                    'objects': _bytes_feature(tf.io.serialize_tensor(annotation['objects']).numpy())
                }))

                tfwriter.write(example.SerializeToString())
        print(f'[+] Written {filename}')
