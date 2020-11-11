import tensorflow as tf

from .data import batch_tensor
from .bbox import get_iou_map, get_regressor_deltas

def classifier_loss(p_true, p_pred):
    """
        The classifier loss explained
        This is basically binary cross entropy
        Args:
            p_true, the true propabilities
            p_pred, the predicted probabilities
        Returns:
            the loss as a scalar
    """
    valid_indices = tf.where(tf.not_equal(p_true, -1.0))
    p_true_valid = tf.gather_nd(p_true, valid_indices)
    p_pred_valid = tf.gather_nd(p_pred, valid_indices)
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
    return binary_crossentropy(p_true_valid, p_pred_valid)

def regressor_loss(t_true, t_pred):
    """
        The regressor loss
        This is smooth L1 loss calculated only on positive anchors
        Args:
            t_true, the true regressor values
            t_pred, the predicted regressor values
        Returns:
            the loss as a scalar
    """
    smooth_l1 = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    batch_size = tf.shape(t_pred)[0]
    t_true = tf.reshape(t_true, [batch_size, -1, 4])
    t_pred = tf.reshape(t_pred, [batch_size, -1, 4])
    loss = smooth_l1(t_true, t_pred)
    valid = tf.math.reduce_any(tf.not_equal(t_true, 0.0), axis=-1)
    valid = tf.cast(valid, tf.float32)
    loss = tf.reduce_sum(loss * valid, axis=-1) # loss vector for each batch
    total_pos_boxes = tf.math.maximum(1.0, tf.reduce_sum(valid, axis=-1))
    return tf.math.reduce_mean(tf.truediv(loss, total_pos_boxes))


def randomly_select_n_from_mask(mask, n):
    """
        Select n True values from the given boolean mask
        Args:
            mask, a batch of boolean mask shape (batch_size, total_elements)
            n,    a tensor of shape (batch_size, ) containing the number of True
                  values to randomly select from the mask in each batch
        Returns:
            a batch of boolean mask whose number of True elements is n or less
            if there are less then n True values in the original mask
    """
    # This genius code is taken from:
    # https://github.com/FurkanOM/tf-rpn/blob/master/utils/train_utils.py#L50
    # and is licensed under the terms of the APACHE public license
    max = tf.math.reduce_max(n) * 10
    random_tensor = tf.random.uniform(tf.shape(mask), minval=1, maxval=max, dtype=tf.int32)
    random_tensor = tf.multiply(random_tensor, tf.cast(mask, tf.int32))
    indices = tf.argsort(random_tensor, direction='DESCENDING')
    sorted_indices = tf.argsort(indices)
    selected_mask = tf.less(sorted_indices, tf.expand_dims(n, 1))
    return tf.math.logical_and(selected_mask, mask)

def get_target(anchors, gt_boxes, config):
    """
        Returns the RPN target for a batch of ground truth boxes
        Args:
            anchors,  a anchor tensor of shape (output_height * output_width * anchor_count, 4)
            gt_boxes, a ground truth box tensor of shape (batch_size, total_gt_boxes, 4)
            config,    the configuration dictionary
        Returns:
            labels,           the output of the RPN classifier
            regressor_output, the output of the RPN regressor
    """
    batch_size = tf.shape(gt_boxes)[0]
    total_gt_boxes = tf.shape(gt_boxes)[1]
    output_height, output_width = config['output_size']
    anchor_count = config['anchor_count']
    region_count = config['region_count']
    variances = config['variances']

    anchors_batch = batch_tensor(anchors, batch_size)
    iou_map = get_iou_map(anchors_batch, gt_boxes) # shape (batch_size, total_anchors, total_gt_boxes)

    # contains at position (batch_idx, gt_idx) the index of the best anchor
    # for the gt box indexed by gt_idx in the batch indexed by batch_idx
    max_anchor_idx_foreach_gt = tf.argmax(iou_map, axis=1, output_type=tf.int32)

    # contains at position (batch_idx, anchor_idx) the index of the best gt box
    # for the anchor indexed by anchor_idx in the batch indexed by batch_id
    max_gt_idx_foreach_anchor = tf.argmax(iou_map, axis=2, output_type=tf.int32)

    batch_indices = tf.range(0, batch_size)
    batch_indices = tf.expand_dims(batch_indices, 1)
    batch_indices = tf.repeat(batch_indices, total_gt_boxes, axis=1)
    best_anchor_for_gt_indices = tf.stack([batch_indices, max_anchor_idx_foreach_gt], axis=-1)
    best_anchor_for_gt_indices = tf.reshape(best_anchor_for_gt_indices, [-1, 2])
    indices_count = tf.shape(best_anchor_for_gt_indices)[0]

    best_iou_for_each_anchor = tf.reduce_max(iou_map, axis=2) # shape (batch_size, total_anchors)

    # a bool tensor containing True at the index (batch_idx, anchor_idx) where an
    # anchor is positive
    positive_anchors = tf.greater(best_iou_for_each_anchor, config['high_threshold'])
    negative_anchors = tf.less(best_iou_for_each_anchor, config['low_threshold'])

    best_anchors = tf.scatter_nd(best_anchor_for_gt_indices, tf.fill((indices_count,), True), tf.shape(positive_anchors))
    positive_anchors = tf.math.logical_or(positive_anchors, best_anchors)
    positive_anchors = randomly_select_n_from_mask(positive_anchors, [region_count])

    positive_count = tf.reduce_sum(tf.cast(positive_anchors, tf.int32), axis=-1)
    negative_count = (region_count * 2) - positive_count

    negative_anchors = randomly_select_n_from_mask(negative_anchors, negative_count)

    labels = tf.where(positive_anchors, 1.0, -1.0)
    labels = tf.add(labels, tf.cast(negative_anchors, tf.float32))
    # in the unlikely event that an anchor whose iou is less than 0.3
    # but its the best fitting anchor for any ground truth box we need to
    # clip possible value of 2.0 to 1.0
    labels = tf.clip_by_value(labels, -1.0, 1.0)

    # find the best fitting gt box for each anchor
    # this tensor has shape (batch_size, total_anchors, 4)
    gt_boxes_foreach_anchor = tf.gather(gt_boxes, max_gt_idx_foreach_anchor, batch_dims=1)
    # replace non positve anchors with zeros
    gt_boxes_foreach_anchor = tf.where(tf.expand_dims(positive_anchors, -1), gt_boxes_foreach_anchor, 0.0)

    deltas = get_regressor_deltas(anchors, gt_boxes_foreach_anchor) / variances

    labels = tf.reshape(labels, [batch_size, output_height, output_width, anchor_count])
    deltas = tf.reshape(deltas, [batch_size, output_height, output_width, anchor_count * 4])
    return labels, deltas
