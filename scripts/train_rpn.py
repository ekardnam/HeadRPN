import sys
sys.path.insert(0, '.')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import argparse

from head_rpn.bbox import (
    generate_anchors,
    normalize_bboxes
)
from head_rpn.config import get_configuration
from head_rpn.data import process_data
from head_rpn.model import get_model
from head_rpn.train import (
    get_target,
    classifier_loss,
    regressor_loss
)

def parse_example_proto(example_proto):
    features = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'image_width': tf.io.FixedLenFeature((), tf.int64),
        'image_height': tf.io.FixedLenFeature((), tf.int64),
        'object_count': tf.io.FixedLenFeature((), tf.int64),
        'objects': tf.io.FixedLenFeature((), tf.string)
    }
    return tf.io.parse_single_example(example_proto, features)

def parse_example_to_data(example, config, is_training):
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
    return image, get_target(anchors, gt_boxes, config)

def get_padded_shapes():
    return ((None, None, None), (None, None))

def create_data_pipeline(filenames, config, is_training=True):
    AUTO = tf.data.experimental.AUTOTUNE
    anchors = generate_anchors(config)
    example_protos = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    examples = example_protos.map(parse_example_proto, num_parallel_calls=AUTO)
    data = examples.map(lambda example: parse_example_to_data(example, config, is_training), num_parallel_calls=AUTO)
    data = data.padded_batch(config['batch_size'], padded_shapes=get_padded_shapes())
    data = data.map(lambda image, gt_boxes: parse_data_to_target(image, gt_boxes, anchors, config))
    return data.prefetch(AUTO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains an RPN over a TFRecord dataset')
    parser.add_argument('wildcard_train', help='The wildcard to find TFRecord files to read for training')
    parser.add_argument('wildcard_validation', help='The wildcard to find TFRecord files to read for validation')
    parser.add_argument('--out', type=str, default='out', help='The output directory for checkpoints and weights')
    parser.add_argument('--logs', type=str, default='logs', help='The output directory for summaries')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--steps', type=int, default=-1, help='Number of train steps per epoch')
    parser.add_argument('--val-steps', type=int, default=-1, help='Number of validation steps per epoch')

    args = parser.parse_args()

    if args.steps == -1 or args.val_steps == -1:
        print('Automatic calculation of steps per epoch is not implemented ¯\_( )_/¯')
        exit(0)

    train_filenames = tf.data.Dataset.list_files(args.wildcard_train)
    val_filenames = tf.data.Dataset.list_files(args.wildcard_validation)

    config = get_configuration()
    train_ds = create_data_pipeline(train_filenames, config)
    val_ds = create_data_pipeline(val_filenames, config, is_training=False)

    model = get_model(config)
    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=[classifier_loss, regressor_loss]
          )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out, 'rpn-{epoch:02d}.h5'))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=args.logs)

    model.fit(
            train_ds,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            validation_data=val_ds,
            validation_steps=args.val_steps,
            callbacks=[
                checkpoint_cb,
                tensorboard_cb
            ]
          )
