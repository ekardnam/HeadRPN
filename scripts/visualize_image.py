import sys
sys.path.insert(0, '.')

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from head_rpn.dataset import (
    load_kaggle_annotations,
    load_xml_annotations,
    aggregate_annotations
)
from head_rpn.draw import draw_image_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize an image from a Kaggle dataset in TensorBoard')
    parser.add_argument('labels', type=str, help='Path to the labels.csv file')
    parser.add_argument('boxes', type=str, help='Path to the boxes.csv file')
    parser.add_argument('scut_annotations_folder', type=str, help='The folder containing SCUT annotations')
    parser.add_argument('kaggle_basepath', type=str, help='The base path for the Kaggle images')
    parser.add_argument('scut_basepath', type=str, help='The base path for SCUT images')
    parser.add_argument('--id', type=int, default=0, help='ID of the image to visualize')
    parser.add_argument('--out', type=str, default='logs', help='The output directory for summaries')

    args = parser.parse_args()
    # load the kaggle annotations
    kaggle_annotations = load_kaggle_annotations(args.labels, args.boxes)
    scut_annotations = load_xml_annotations(args.scut_annotations_folder)

    annotations = aggregate_annotations(
                                        [kaggle_annotations, scut_annotations],
                                        [args.kaggle_basepath, args.scut_basepath]
                                       )

    # process the first image
    image_annotation = annotations[args.id]

    image = tf.image.decode_jpeg(tf.io.read_file(image_annotation["image_filename"]))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (480, 640))
    image_batch = tf.expand_dims(image, axis=0)

    writer = tf.summary.create_file_writer(args.out)
    with writer.as_default():
        draw_image_batch('Image', image_batch)

    writer.flush()
