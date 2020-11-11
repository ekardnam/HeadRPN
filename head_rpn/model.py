import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

def get_model(config, base_model_trainable=False):
    """
        Create the RPN model from the config
        Args:
            config,                the configuration
            base_model_trainable, whether to train the base model (defaults to False)
        Returns:
            the Keras RPN model
    """
    image_height, image_width = config['image_height'], config['image_width']
    base_model = VGG16(include_top=False, input_shape=(image_height, image_width, 3))
    base_model.trainable = base_model_trainable
    vgg16_feature_map = base_model.get_layer('block5_conv3').output
    output = Conv2D(512, (3, 3), activation='relu', padding='same', name='RPN_conv')(vgg16_feature_map)
    rpn_class = Conv2D(config['anchor_count'], (1, 1), activation='sigmoid', name='RPN_classifier')(output)
    rpn_regr = Conv2D(config['anchor_count'] * 4, (1, 1), activation='linear', name='RPN_regressor')(output)
    return Model(inputs=base_model.input, outputs=[rpn_class, rpn_regr])
