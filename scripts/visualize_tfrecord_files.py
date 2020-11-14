import sys
sys.path.insert(0, '.')

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from head_rpn.draw import draw_image_batch, draw_bounding_boxes
from head_rpn.bbox import normalize_bboxes, convert_bounding_boxes_to_tf_format

def parse_example_proto(example_proto):
    features = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'image_width': tf.io.FixedLenFeature((), tf.int64),
        'image_height': tf.io.FixedLenFeature((), tf.int64),
        'object_count': tf.io.FixedLenFeature((), tf.int64),
        'objects': tf.io.FixedLenFeature((), tf.string)
    }
    return tf.io.parse_single_example(example_proto, features)

def get_resized_image(example):
    return tf.image.resize(
            tf.image.convert_image_dtype(tf.image.decode_jpeg(example['image']), tf.float32),
            (480, 640)
           )

def get_ground_truth_boxes(example):
    return normalize_bboxes(
            tf.io.parse_tensor(example['objects'], tf.int32),
            tf.cast(example['image_height'], tf.float32),
            tf.cast(example['image_width'], tf.float32)
           )

def parse_example(example):
    return get_resized_image(example), get_ground_truth_boxes(example)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizes the contents of TFRecord files in TensorBoard')
    parser.add_argument('wildcard', help='The wildcard to find TFRecord files to read')
    parser.add_argument('--out', type=str, default='logs', help='The output directory for summaries')
    parser.add_argument('--batch', type=int, default=100, help='The batch size of each image batch to display')
    parser.add_argument('--max-imgs', type=int, default=3, help='The numbers of images to display in each batch')
    args = parser.parse_args()

    AUTO = tf.data.experimental.AUTOTUNE
    filenames = tf.data.Dataset.list_files(args.wildcard)
    example_protos = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    examples = example_protos.map(parse_example_proto, num_parallel_calls=AUTO)

    data = examples.map(parse_example, num_parallel_calls=AUTO)
    data = data.padded_batch(args.batch, padded_shapes=((None, None, None), (None, None)))
    writer = tf.summary.create_file_writer(args.out)
    i = 0
    for image_batch, gt_batch in data:
        print(f'[+] Processing batch {i}')
        image_batch = draw_bounding_boxes(image_batch, convert_bounding_boxes_to_tf_format(gt_batch))
        with writer.as_default():
            draw_image_batch(f'Batch {i}', image_batch, max_images=args.max_imgs)
        i += 1

    writer.flush()
