import sys
sys.path.insert(0, '.')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import argparse

from head_rpn.model import get_model
from head_rpn.config import get_configuration
from head_rpn.bbox import (
    get_bounding_boxes_from_labels,
    apply_deltas_to_bounding_boxes,
    convert_bounding_boxes_to_tf_format
)
from head_rpn.draw import (
    draw_bounding_boxes,
    draw_image_batch
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uses a model to make a prediction on a single image')
    parser.add_argument('model_path', type=str, help='The path to the h5 file with model weights')
    parser.add_argument('image_path', type=str, help='The path to the image')
    parser.add_argument('--out', type=str, default='logs', help='TensorBoard log folder')
    args = parser.parse_args()

    config = get_configuration()

    image = tf.image.decode_jpeg(tf.io.read_file(args.image_path))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [config['image_height'], config['image_width']])
    image_batch = tf.expand_dims(image, axis=0)

    model = get_model(config)
    model.load_weights(args.model_path)

    labels, deltas = model.predict(image_batch)
    bboxes = get_bounding_boxes_from_labels(labels, config)
    bboxes = apply_deltas_to_bounding_boxes(bboxes, tf.reshape(deltas, [-1, 4]), config)
    bboxes = convert_bounding_boxes_to_tf_format(bboxes)

    bboxes = tf.squeeze(bboxes, [0]) # apparently after all the efforts done working with bbox batches
                                     # tensorflow betrayed us and decided that nms should work on
                                     # non batched boxes. lame
    selected_indices = tf.image.non_max_suppression(bboxes, tf.reshape(labels, [-1]), tf.shape(bboxes)[0], iou_threshold=0.3)
    selected_boxes = tf.gather(bboxes, selected_indices)

    selected_boxes = tf.expand_dims(selected_boxes, axis=0) # now tensorflow wants the batches again
                                                            # so lame
    output_image = draw_bounding_boxes(image_batch, selected_boxes)

    writer = tf.summary.create_file_writer(args.out)
    with writer.as_default():
        draw_image_batch('Predicted boxes', output_image)

    writer.flush()
