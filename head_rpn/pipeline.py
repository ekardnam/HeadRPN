import tensorflow as tf

from .bbox import (
    generate_anchors,
    normalize_bboxes
)
from .data import process_data
from .train import get_target

def parse_example_proto(example_proto):
    """
        Parses and example proto to the example object
    """
    features = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'image_width': tf.io.FixedLenFeature((), tf.int64),
        'image_height': tf.io.FixedLenFeature((), tf.int64),
        'object_count': tf.io.FixedLenFeature((), tf.int64),
        'objects': tf.io.FixedLenFeature((), tf.string)
    }
    return tf.io.parse_single_example(example_proto, features)

def parse_example_to_data(example, config, is_training):
    """
        Parses and example to data
        Args:
            example,     the example
            config,       the configuation dictionary
            is_training, whether its data for training (True) or validation (False)
        Returns:
            image,    the image tensor
            gt_boxes, the ground truth boxes tensor
    """
    image = tf.image.decode_jpeg(example['image'])
    gt_boxes = tf.io.parse_tensor(example['objects'], out_type=tf.int32)
    gt_boxes = normalize_bboxes(
                gt_boxes,
                tf.cast(example['image_height'], tf.float32),
                tf.cast(example['image_width'], tf.float32)
               )
    image, gt_boxes = process_data(image, gt_boxes, config['image_height'], config['image_width'], apply_augmentation=is_training)
    return image, gt_boxes

def parse_data_to_target(image, gt_boxes, anchors, config):
    """
        Applies get_target to data entries
        Args:
            image,    the image tensor
            gt_boxes, the gt boxes tensor
            anchors,  the anchors tensor
            config,    the configuration dictionary
        Returns:
            image,  the image tensor
            target, the target of the model
    """
    return image, get_target(anchors, gt_boxes, config)

def get_padded_shapes():
    """
        Returns the padded shapes for the padded_batch
    """
    return ((None, None, None), (None, None))

def create_data_pipeline(filenames, config, is_training=True):
    """
        Creates the data pipeline from TFRecord files
        Args:
            filenames,    a tf.data.Dataset containing the TFRecord filenames
            config,       the configuration dictionary
            is_training, whether it is a pipeline for training
        Returns:
            the tf.data.Dataset pipeline
    """
    AUTO = tf.data.experimental.AUTOTUNE
    anchors = generate_anchors(config)
    example_protos = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    examples = example_protos.map(parse_example_proto, num_parallel_calls=AUTO)
    data = examples.map(lambda example: parse_example_to_data(example, config, is_training), num_parallel_calls=AUTO)
    data = data.padded_batch(config['batch_size'], padded_shapes=get_padded_shapes())
    data = data.map(lambda image, gt_boxes: parse_data_to_target(image, gt_boxes, anchors, config))
    return data.prefetch(AUTO)
