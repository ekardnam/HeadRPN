import sys
sys.path.insert(0, '.')

import tensorflow as tf

from head_rpn.dataset import load_kaggle_annotations
from head_rpn.train import get_target
from head_rpn.draw import draw_bounding_boxes, draw_image_batch
from head_rpn.bbox import (
    get_bounding_boxes_from_classifier_output,
    generate_anchors,
    normalize_bboxes
)
from head_rpn.config import get_configuration

# load the kaggle annotations
annotations = load_kaggle_annotations('dataset/labels.csv', 'dataset/boxes.csv')

# process the first image
image_annotation = annotations[2000]

image = tf.image.decode_jpeg(tf.io.read_file(f'dataset/images/images/{image_annotation["image_filename"]}'))
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

classifier_output, _ = get_target(anchors, gt_boxes, config)
classifier_output = tf.squeeze(classifier_output, [0]) # unbatch
predicted_boxes = get_bounding_boxes_from_classifier_output(classifier_output, config)
predicted_boxes = tf.expand_dims(predicted_boxes, axis=0) # rebatch
image_batch = draw_bounding_boxes(image_batch, predicted_boxes)
writer = tf.summary.create_file_writer('logs')
with writer.as_default():
    draw_image_batch('Image', image_batch)
    writer.flush()
