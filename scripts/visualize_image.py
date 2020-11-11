import sys
sys.path.insert(0, '.')

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from head_rpn.dataset import load_kaggle_annotations
from head_rpn.draw import draw_image_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize an image from a Kaggle dataset in TensorBoard')
    parser.add_argument('labels', type=str, help='Path to the labels.csv file')
    parser.add_argument('boxes', type=str, help='Path to the boxes.csv file')
    parser.add_argument('basepath', type=str, help='The base path for the images in the dataset')
    parser.add_argument('--id', type=int, default=0, help='ID of the image to visualize')
    parser.add_argument('--out', type=str, default='logs', help='The output directory for summaries')

    args = parser.parse_args()
    # load the kaggle annotations
    annotations = load_kaggle_annotations(args.labels, args.boxes)

    # process the first image
    image_annotation = annotations[args.id]

    image = tf.image.decode_jpeg(tf.io.read_file(os.path.join(args.basepath, image_annotation["image_filename"])))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (480, 640))
    image_batch = tf.expand_dims(image, axis=0)

    writer = tf.summary.create_file_writer(args.out)
    with writer.as_default():
        draw_image_batch('Image', image_batch)

    writer.flush()
