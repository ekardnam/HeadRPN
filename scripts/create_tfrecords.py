import sys
sys.path.insert(0, '.')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import argparse

from head_rpn.dataset import load_kaggle_annotations
from math import ceil

TFRECORD_IMG_COUNT = 200

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
    parser.add_argument('basepath', type=str, help='The base path for the images in the dataset')
    parser.add_argument('--out', type=str, default='out/data-{}.tfrecord', help='Format string for output TFRecords')
    parser.add_argument('--img-count', type=int, default=TFRECORD_IMG_COUNT, help='The number of images in each TFRecord file')

    args = parser.parse_args()

    annotations = load_kaggle_annotations(args.labels, args.boxes)
    annotation_count = len(annotations)

    for i in range(ceil(annotation_count / TFRECORD_IMG_COUNT)):
        filename = args.out.format(i)
        annotations_to_process = annotations[i * TFRECORD_IMG_COUNT : (i + 1) * TFRECORD_IMG_COUNT]
        with tf.python.python_io.TFRecordWriter(args.out.format(i)) as tfwriter:
            for annotation in annotations_to_process:
                image = tf.io.read_file(os.path.join(args.basepath, annotation["image_filename"]))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(image.numpy()),
                    'image_width': _int64_feature(annotation['image_width']),
                    'image_height': _int64_feature(annotation['image_height']),
                    'object_count': _int64_feature(annotation['object_count']),
                    'objects': _bytes_feature(tf.io.serialize_tensor(annotation['objects']).numpy())
                }))

                tfwriter.write(example.SerializeToString())
        print(f'[+] Written {filename}')
