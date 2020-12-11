import tensorflow as tf

def batch_tensor(tensor, batch_size):
    """
        Batches a tensor
        Args:
            tensor,     a tensor of shape *
            batch_size, the batch_size
        Returns:
            a tensor of shape (batch_size, *) witht the tensor repeated in each
            batch entry
    """
    batch = tf.expand_dims(tensor, axis=0)
    return tf.repeat(batch, batch_size, axis=0)

def get_random_bool():
    """
        Generates a random bool scalar tensor
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

def horizontal_flip(image, gt_boxes):
    """
        Flips the image horizontally
        Args:
            image,    the image tensor
            gt_boxes, the normalized ground truth boxes tensor
        Returns:
            image,    flipped image
            gt_boxes, flipped gt_boxes
    """
    flipped_image = tf.image.flip_left_right(image)
    flipped_gt_boxes = tf.stack(
        [
            1.0 - gt_boxes[..., 2],
            gt_boxes[..., 1],
            1.0 - gt_boxes[..., 0],
            gt_boxes[..., 3]
        ], axis=-1
    )
    return flipped_image, flipped_gt_boxes

def gaussian_noise(image, gt_boxes):
    """
        Applies color modifications to the images
        Args:
             image, image tensor
             gt_boxes, normalized gt_boxes
        Returns:
             noised_image, color modified image tensor
             gt_boxes, normalized gt_boxes
    """
    color_noise = tf.random.normal(tf.shape(image), 0.0,  0.1, tf.float32)
    noised_image = tf.add(image, color_noise)
    return tf.clip_by_value(noised_image, 0.0, 1.0), gt_boxes

def process_data(image, gt_boxes, height, width, apply_augmentation=False):
    """
        Data processing operation
        Args:
            image,              the image
            gt_boxes,           the ground truth boxes
            height,             the height to resize the image data to
            width,              the width to resize the image data to
            apply_augmentation, whether to apply data augmentation
        Returns:
            image, a tensor containing the image of shape (height, width, channels)
            gt_boxes, a tensor containing normalized ground truth boxes of
            shape (gt_boxes_count, 4 [x1, y1, x2, y2])
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (height, width))
    if apply_augmentation:
        image, gt_boxes = tf.cond(get_random_bool(), lambda: horizontal_flip(image, gt_boxes), lambda: (image, gt_boxes))
        image, gt_boxes = tf.cond(get_random_bool(), lambda: gaussian_noise(image, gt_boxes), lambda: (image, gt_boxes))
    return image, gt_boxes
