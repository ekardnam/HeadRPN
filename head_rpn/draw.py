import tensorflow as tf

from .bbox import convert_bounding_boxes_to_tf_format

def draw_image_batch(name, image_batch, max_images=3):
    """
        Draws a image batch using TensorBoard
        Args:
            name,        a name for the resulting TensorBoard
            image_batch, the image batch to draw
            max_images,  the max amount of images to display
    """
    tf.summary.image(name, image_batch, max_outputs=max_images, step=0)

def draw_bounding_boxes(image_batch, bboxes_batch, color=[1, 0, 0, 1]):
    """
        Draws a batch of bounding boxes on a batch of images
        Args:
            image_batch,  a tensor of shape (batch_size, height, width, channels)
            bboxes_batch, a tensor of shape (batch_size, total_bboxes, 4) containing normalized
                          bounding boxes
            color,        the color in the form of a list of RGBA values normalized to 1
                          defaults to red ([1, 0, 0, 1])
        Returns:
            a batch of images with the corresponding bboxes drawn
    """
    colors = tf.constant([color], dtype=tf.float32)
    return tf.image.draw_bounding_boxes(image_batch, convert_bounding_boxes_to_tf_format(bboxes_batch), colors)
