import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import tensorflow as tf

from test_bbox import BboxTestCase
from test_data import DataTestCase

if __name__ == '__main__':
    tf.test.main()
