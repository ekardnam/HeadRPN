import tensorflow as tf

from head_rpn.data import horizontal_flip

class DataTestCase(tf.test.TestCase):
    def test_horizontal_flip(self):
        image = tf.constant([
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]],

            [[7.0, 8.0, 9.0],
             [10.0, 11.0, 12.0]]
        ], dtype=tf.float32)
        gt_boxes = tf.constant([
            [0.3, 0.4, 0.6, 0.6]
        ], dtype=tf.float32)

        real_flipped_image = tf.constant([
            [[4.0, 5.0, 6.0],
             [1.0, 2.0, 3.0]],

            [[10.0, 11.0, 12.0],
             [7.0, 8.0, 9.0]]
        ], dtype=tf.float32)
        real_flipped_gt_boxes = tf.constant([
            [0.4, 0.4, 0.7, 0.6]
        ], dtype=tf.float32)
        flipped_image, flipped_gt_boxes = horizontal_flip(image, gt_boxes)
        self.assertAllClose(flipped_image, real_flipped_image)
        self.assertAllClose(flipped_gt_boxes, real_flipped_gt_boxes)
