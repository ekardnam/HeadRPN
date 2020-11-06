import tensorflow as tf

from .data import batch_tensor
from .bbox import get_iou_map

def get_target(anchors, gt_boxes, config):
    """
        Returns the RPN target for a batch of ground truth boxes
        Args:
            anchors, a tensor of shape (output_height * output_width * anchor_count, 4)
            gt_boxes, a tensor of shape (batch_size, total_gt_boxes, 4)
            config, the configuration dictionary
        Returns:
            classifier_output, the output of the RPN classifier
            regressor_output, the output of the RPN regressor
    """
    batch_size = tf.shape(gt_boxes)[0]
    total_gt_boxes = tf.shape(gt_boxes)[1]
    output_height, output_width = config['output_size']
    anchor_count = config['anchor_count']
    region_count = config['region_count']

    anchors_batch = batch_tensor(anchors, batch_size)
    iou_map = get_iou_map(anchors_batch, gt_boxes) # shape (batch_size, total_anchors, total_gt_boxes)

    # contains at position (batch_idx, gt_idx) the index of the best anchor
    # for the gt box indexed by gt_idx in the batch indexed by batch_idx
    max_idx_for_each_gt = tf.argmax(iou_map, axis=1, output_type=tf.int32)

    batch_indices = tf.range(0, batch_size)
    batch_indices = tf.expand_dims(batch_indices, 1)
    batch_indices = tf.repeat(batch_indices, total_gt_boxes, axis=1)
    best_anchor_for_gt_indices = tf.stack([batch_indices, max_idx_for_each_gt], axis=2)
    best_anchor_for_gt_indices = tf.reshape(best_anchor_for_gt_indices, [-1, 2])
    indices_count = tf.shape(best_anchor_for_gt_indices)[0]

    best_iou_for_each_anchor = tf.reduce_max(iou_map, axis=2) # shape (batch_size, total_anchors)

    # a bool tensor containing True at the index (batch_idx, anchor_idx) where an
    # anchor is positive
    positive_anchors = tf.greater(best_iou_for_each_anchor, config['high_threshold'])

    best_anchors = tf.scatter_nd(best_anchor_for_gt_indices, tf.fill((indices_count,), True), tf.shape(positive_anchors))
    positive_anchors = tf.math.logical_or(positive_anchors, best_anchors)

    classifier_output = tf.reshape(positive_anchors, [batch_size, output_height, output_width, anchor_count])
    return tf.cast(classifier_output, tf.float32), tf.constant(0.0)
