import sys
sys.path.insert(0, '.')

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from head_rpn.dataset import load_kaggle_annotations
from head_rpn.train import get_target
from head_rpn.draw import draw_bounding_boxes, draw_image_batch
from head_rpn.bbox import (
    get_bounding_boxes_from_labels,
    generate_anchors,
    normalize_bboxes,
    apply_deltas_to_bounding_boxes
)
from head_rpn.config import get_configuration

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

    # use this configuration configuration
    CONFIGURATION_PROTO = {

        'image_width': 640,
        'image_height': 480,

        'base_model_stride': 16,

        'anchor_scales': [32, 64, 128],
        'anchor_ratios': [1],

        'batch_size': 1,
        'region_count': 32,

        'high_threshold': 0.7,
        'low_threshold': 0.3

    }
    config = get_configuration(config_proto=CONFIGURATION_PROTO)

    anchors = generate_anchors(config)
    gt_boxes = tf.constant([image_annotation['objects']])
    gt_boxes = normalize_bboxes(gt_boxes, image_annotation['image_height'], image_annotation['image_width'])

    labels, deltas = get_target(anchors, gt_boxes, config)
    predicted_anchors = get_bounding_boxes_from_labels(labels, config)
    predicted_boxes = apply_deltas_to_bounding_boxes(predicted_anchors, tf.reshape(deltas, [1, -1, 4]))
    image_batch = draw_bounding_boxes(image_batch, predicted_boxes)
    writer = tf.summary.create_file_writer(args.out)
    with writer.as_default():
        draw_image_batch('Image', image_batch)

    writer.flush()
