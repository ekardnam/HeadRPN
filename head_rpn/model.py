import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

def get_model(config):
    """
        Create the RPN model from the config
        Args:
            config,  the configuration dictionary
        Returns:
            the Keras RPN model
    """
    image_height, image_width = config['image_height'], config['image_width']
    base_model = VGG16(include_top=False, input_shape=(image_height, image_width, 3))
    vgg16_feature_map = base_model.get_layer('block5_conv3').output
    output = Conv2D(512, (3, 3), activation='relu', padding='same', name='RPN_conv')(vgg16_feature_map)
    rpn_class = Conv2D(config['anchor_count'], (1, 1), activation='sigmoid', name='RPN_classifier')(output)
    rpn_regr = Conv2D(config['anchor_count'] * 4, (1, 1), activation='linear', name='RPN_regressor')(output)
    return Model(inputs=base_model.input, outputs=[rpn_class, rpn_regr])

def get_regularized_model(config):
    """
        Creates the RPN model from the config using L2 regularization
        Args:
            config,  the configuration dictionary
        Returns:
            the Keras RPN model
    """
    image_height, image_width = config['image_height'], config['image_width']
    base_model = VGG16(include_top=False, input_shape=(image_height, image_width, 3))
    vgg16_feature_map = base_model.get_layer('block5_conv3').output
    output = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer='l2', name='RPN_conv')(vgg16_feature_map)
    rpn_class = Conv2D(config['anchor_count'], (1, 1), activation='sigmoid', name='RPN_classifier')(output)
    rpn_regr = Conv2D(config['anchor_count'] * 4, (1, 1), activation='linear', name='RPN_regressor')(output)
    return Model(inputs=base_model.input, outputs=[rpn_class, rpn_regr])

def get_model_for_tuning(name, custom_obj):
    model = load_model(name, custom_obj)
    for layer in model.layers:
        if (layer.name == 'block5_conv1' or layer.name == 'block5_conv2' 
           or layer.name == 'block5_conv3' or layer.name == 'RPN_conv' or layer.name == 'RPN_classifier'
           or layer.name == 'RPN_regressor'):
              layer.trainable = True
           else:
              layer.trainable = False
    print(len(model.trainable_weights))
    print(len(model.non_trainable_weights))  
    return model
