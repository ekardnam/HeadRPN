import tensorflow as tf

from head_rpn.bbox import (
    generate_base_anchors,
    broadcast_base_anchors_to_grid,
    cartesian_product,
    generate_coordinate_grid,
    generate_anchors,
    get_regressor_deltas,
    get_iou_map
)
from head_rpn.config import get_configuration

TEST_CONFIGURATION_PROTO = {
    'image_width': 32,
    'image_height': 32,

    'base_model_stride': 16,

    'anchor_scales': [32, 64],
    'anchor_ratios': [1, 2]
}

class BboxTestCase(tf.test.TestCase):
    def setUp(self):
        self.test_config = get_configuration(TEST_CONFIGURATION_PROTO)
        self.real_base_anchors = tf.constant([
            [-16, -16, 16, 16],
            [-16, -32, 16, 32],
            [-32, -32, 32, 32],
            [-32, -64, 32, 64]
        ], dtype=tf.float32) / tf.constant([[
            self.test_config['image_width'],
            self.test_config['image_height'],
            self.test_config['image_width'],
            self.test_config['image_height']
        ]], dtype=tf.float32)

    def test_generate_base_anchors(self):
        base_anchors = generate_base_anchors(self.test_config)
        self.assertAllClose(base_anchors, self.real_base_anchors)

    def test_broadcast_base_anchors_to_grid_shape(self):
        broadcasted_anchors = broadcast_base_anchors_to_grid(self.real_base_anchors, 10, 20)
        self.assertAllEqual(broadcasted_anchors.shape, tf.constant([10, 20, 4, 4]))

    def test_cartesian_product(self):
        a = tf.constant([1, 2])
        b = tf.constant([3, 4, 5])
        result = tf.constant([
            [[1, 3], [2, 3]],
            [[1, 4], [2, 4]],
            [[1, 5], [2, 5]]
        ])
        product = cartesian_product(a, b, 2, 3)
        self.assertAllClose(product, result)

    def test_generate_coordinate_grid(self):
        coord_grid = generate_coordinate_grid(2, 2, 1)
        real_coord_grid = tf.constant(
            [[[[0.25, 0.25, 0.25, 0.25]],
              [[0.75, 0.25, 0.75, 0.25]]],
             [[[0.25, 0.75, 0.25, 0.75]],
              [[0.75, 0.75, 0.75, 0.75]]]]
        )
        self.assertAllClose(coord_grid, real_coord_grid)

    def test_generate_anchors_shape(self):
        anchors_shape = generate_anchors(self.test_config).shape
        output_height, output_width = self.test_config['output_size']
        real_shape = tf.constant([output_height * output_width * self.test_config['anchor_count'], 4])
        self.assertAllEqual(anchors_shape, real_shape)

    def test_generate_anchors(self):
        anchors = generate_anchors(self.test_config)
        real_anchors = tf.constant(
            [[0.,   0.,   0.75, 0.75],
             [0.,   0.,   0.75, 1.  ],
             [0.,   0.,   1.,   1.  ],
             [0.,   0.,   1.,   1.  ],
             [0.25, 0.,   1.,   0.75],
             [0.25, 0.,   1.,   1.  ],
             [0.,   0.,   1.,   1.  ],
             [0.,   0.,   1.,   1.  ],
             [0.,   0.25, 0.75, 1.  ],
             [0.,   0.,   0.75, 1.  ],
             [0.,   0.,   1.,   1.  ],
             [0.,   0.,   1.,   1.  ],
             [0.25, 0.25, 1.,   1.  ],
             [0.25, 0.,   1.,   1.  ],
             [0.,   0.,   1.,   1.  ],
             [0.,   0.,   1.,   1.  ]]
        )
        self.assertAllClose(anchors, real_anchors)

    def test_get_regressor_deltas(self):
        gt_boxes = tf.constant([
            [.1, .1, .6, .6]
        ])
        bboxes = tf.constant([
            [.05, .05, .61, .61]
        ])
        deltas = get_regressor_deltas(bboxes, gt_boxes)
        real_deltas = tf.constant(
            [[0.03571431,  0.03571431, -0.1133287,  -0.1133287]]
        )
        self.assertAllClose(deltas, real_deltas)

    def test_get_iou_map(self):
        gt_boxes = tf.constant([
            [.1, .1, .6, .6]
        ])
        bboxes = tf.constant([
            [.05, .05, .61, .61]
        ])
        gt_boxes_batch = tf.expand_dims(gt_boxes, 0)
        bboxes_batch = tf.expand_dims(bboxes, 0)
        iou_map = get_iou_map(bboxes_batch, gt_boxes_batch)
        real_iou = tf.constant([[[0.7971939]]])
        self.assertAllClose(iou_map, real_iou)
