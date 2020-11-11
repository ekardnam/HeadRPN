import tensorflow as tf

from .data import batch_tensor

def generate_base_anchors(config):
    """
        Operation that generates the base anchor shapes from the data in configuration
        Args:
            config, the configuration dictionary
        Returns:
            base_anchors, a tensor of shape (anchor_count, 4 [x1, y1, x2, y2])
    """
    image_width = config['image_width']
    image_height = config['image_height']
    anchor_ratios = config['anchor_ratios']
    anchor_scales = config['anchor_scales']
    base_anchors = []
    for width in anchor_scales:
        for ratio in anchor_ratios:
            height = width * ratio
            w = width / image_width   # normalize the anchor width
            h = height / image_height
            base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return tf.cast(base_anchors, dtype=tf.float32)

def broadcast_base_anchors_to_grid(base_anchors, height, width):
    """
        Operation that broadcast the base_anchors tensor tiling it to fill the whole
        RPN output
        Args:
            base_anchors, the base anchor tensor
            height, the height of the RPN output
            width, the width of the RPN output
        Returns:
            base_anchors_grid, tiled version of base_anchors of shape (height, width, anchor_count, 4)
    """
    base_anchors = tf.expand_dims(base_anchors, 0)
    base_anchors = tf.expand_dims(base_anchors, 0)
    return tf.tile(base_anchors, [height, width, 1, 1])

def cartesian_product(x, y, width, height):
    """
        Returns the cartesian product between two vectors
        Args:
            x, a vector of shape (width)
            y, a vector of shape (height)
        Returns:
            x x y, a rank 2 tensor of shape (height, width, 2) containing [i, j, [a_i, b_j]]
    """
    X, Y = tf.meshgrid(x, y) # this has shape (B, A)
    cartesian_product = tf.stack([X, Y], axis=-1)
    return cartesian_product


def generate_coordinate_grid(height, width, anchor_count):
    """
        Generates the normalized coordinates of the rectangle [0, 0] x [width, height]
        into a tensor
        Args:
            height, the height of the rectangle
            width, the width of the rectangle
            anchor_count, the anchor_count
        Returns:
            a tensor of shape (height, width, anchor_count, 4 [x, y, x, y])
    """
    half_pixel_y = 0.5 * (1 / height)
    half_pixel_x = 0.5 * (1 / width)

    grid_coord_y = tf.cast(
        tf.range(0, height) / height + half_pixel_y,
        dtype=tf.float32
    )
    grid_coord_x = tf.cast(
        tf.range(0, width) / width + half_pixel_x,
        dtype=tf.float32
    )
    coordinates = cartesian_product(grid_coord_x, grid_coord_y, width, height)
    coordinates = tf.expand_dims(coordinates, axis=2)
    return tf.tile(coordinates, [1, 1, anchor_count, 2])

def generate_anchor_grid(config):
    """
        Generates the anchors in a grid
        Args:
            config, the configuation dictionary
        Returns:
            anchors, a tensor of shape (output_heigh, output_width, anchor_count, 4 [x1, y1, x2, y2])
                     containing the anchors relative to the output of the RPN
    """
    anchor_count = config['anchor_count']
    output_height, output_width = config['output_size']

    coordinates_grid = generate_coordinate_grid(output_height, output_width, anchor_count)
    base_anchors = generate_base_anchors(config)
    base_anchors_grid = broadcast_base_anchors_to_grid(base_anchors, output_height, output_width)
    return tf.clip_by_value(base_anchors_grid + coordinates_grid, 0, 1)

def generate_anchors(config):
    """
        Generates all the anchors
        Args:
            config, the configuation dictionary
        Returns:
            anchors, a tensor of shape (output_height * output_width * anchor_count, 4 [x1, y1, x2, y2])
                     containing the anchors relative to the output of the RPN
    """
    anchor_grid = generate_anchor_grid(config)
    return tf.reshape(anchor_grid, [-1, 4])

def get_bbox_width_height(bbox):
    width = bbox[..., 2] - bbox[..., 0]
    height = bbox[..., 3] - bbox[..., 1]
    return width, height

def get_bbox_centre(bbox):
    centre_x = bbox[..., 2] + bbox[..., 0]
    centre_y = bbox[..., 3] + bbox[..., 1]
    return 0.5 * centre_x, 0.5 * centre_y

def get_regressor_deltas(bboxes, gt_boxes):
    """
        Takes a batch of total_bboxes bboxes and gt_boxes and returns a batch of regressor deltas
        Args:
            bboxes, a batch of bboxes of shape (batch_size, total_bboxes, 4)
            gt_boxes, a batch of gt_boxes of shape (batch_size, total_bboxes, 4)
        Returns:
            a tensor of shape (batch_size, total_bboxes, 4)
    """
    bbox_width, bbox_height = get_bbox_width_height(bboxes)
    bbox_ctr_x, bbox_ctr_y = get_bbox_centre(bboxes)
    gt_width, gt_height = get_bbox_width_height(gt_boxes)
    gt_ctr_x, gt_ctr_y = get_bbox_centre(gt_boxes)

    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width) # divide by zero
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)

    delta_x = tf.where(tf.equal(gt_width, 0), 0.0, tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), 0.0, tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), 0.0, tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), 0.0, tf.math.log(gt_height / bbox_height))

    return tf.stack([delta_x, delta_y, delta_w, delta_h], axis=-1)

def get_iou_map(bboxes, gt_boxes):
    """
        Takes a batch of total_bboxes bboxes and total_gt_boxes gt_boxes and returns a batch of intersection
        over union maps
        Args:
            bboxes, a batch of bboxes of shape (batch_size, total_bboxes, 4)
            gt_boxes, a batch of gt_boxes of shape (batch_size, total_gt_boxes, 4)
        Returns:
            The IoU map of shape (batch_size, total_bboxes, total_gt_boxes)
    """
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = tf.split(bboxes, 4, axis=-1)
    gt_x1, gt_y1, gt_x2, gt_y2 = tf.split(gt_boxes, 4, axis=-1)

    gt_area = tf.squeeze((gt_x2 - gt_x1) * (gt_y2 - gt_y1), axis=-1)
    bbox_area = tf.squeeze((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1), axis=-1)

    min_x = tf.math.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    min_y = tf.math.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    max_x = tf.math.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    max_y = tf.math.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))

    intersection_area = tf.math.maximum(max_x - min_x, 0) * tf.math.maximum(max_y - min_y, 0)
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    return tf.truediv(intersection_area, union_area)

def denormalize_bboxes(bboxes, height, width):
    """
        Takes a batch of normalized bboxes and denormalizes it
        Args:
            bboxes, a batch of bboxes of shape (batch_size, total_bboxes, 4)
            height, the height
            width,  the width
        Returns:
            the denormalized bboxes in a shape (batch_size, total_bboxes, 4)
    """
    bboxes = tf.cast(bboxes, tf.float32)
    x1 = bboxes[..., 0] * width
    y1 = bboxes[..., 1] * height
    x2 = bboxes[..., 2] * width
    y2 = bboxes[..., 3] * height
    return tf.stack([x1, y1, x2, y2], axis=-1)

def normalize_bboxes(bboxes, height, width):
    """
        Takes a batch of bboxes and normalizes it
        Args:
            bboxes, a batch of bboxes of shape (batch_size, total_bboxes, 4)
            height, the height
            width,  the width
        Returns:
            the normalized bboxes in a shape (batch_size, total_bboxes, 4)
    """
    return denormalize_bboxes(bboxes, 1/height, 1/width)

def get_bounding_boxes_from_labels(labels, config, threshold=0.5):
    """
        Returns a bounding box tensor from the given labels
        Each output is converted into the corresponding bounding box if greater then threshold
        Args:
            labels, the RPN classifier output, has shape (batch_size, output_height, output_width, anchor_count)
            config,            the configuration dictionary
            threshold,        the threshold, defaults to 0.5
        Returns:
            a bounding box tensor of shape (batch_size, total_positive_anchors, 4)
    """
    output_height, output_width = config['output_size']
    anchor_count = config['anchor_count']
    batch_size = tf.shape(labels)[0]
    labels = tf.reshape(labels, [batch_size, -1])
    mask = tf.greater(labels, threshold)
    return tf.where(tf.expand_dims(mask, axis=-1), batch_tensor(generate_anchors(config), batch_size), 0.0)

def apply_deltas_to_bounding_boxes(bboxes, deltas, config):
    """
        Applies the given deltas to the given bounding boxes
        Args:
            bboxes, a tensor containing batches of bounding boxes
                    has shape (batch_size, bbox_count, 4)
            deltas, a tensor containing batches of deltas
                    has shape (batch_size, bbox_count, 4)
            config,  the configuration dictionary
        Returns:
            the batch of bounding boxes with the deltas applied
    """
    deltas = deltas * config['variances']
    width = bboxes[..., 2] - bboxes[..., 0]
    height = bboxes[..., 3] - bboxes[..., 1]
    centre_x = 0.5 * (bboxes[..., 0] + bboxes[..., 2])
    centre_y = 0.5 * (bboxes[..., 1] + bboxes[..., 3])

    result_width = tf.math.exp(deltas[..., 2]) * width
    result_height = tf.math.exp(deltas[..., 3]) * height
    result_centre_x = (deltas[..., 0] * width) + centre_x
    result_centre_y = (deltas[..., 1] * height) + centre_y

    x1 = result_centre_x - result_width * 0.5
    x2 = result_centre_x + result_width * 0.5
    y1 = result_centre_y - result_height * 0.5
    y2 = result_centre_y + result_height * 0.5

    return tf.stack([x1, y1, x2, y2], axis=-1)
