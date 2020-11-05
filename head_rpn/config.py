import copy

DEFAULT_CONFIGURATION_PROTO = {

    'image_width': 640,
    'image_height': 480,

    'base_model_stride': 16,

    'anchor_scales': [32, 64],
    'anchor_ratios': [1]

}

def get_configuration(config_proto=DEFAULT_CONFIGURATION_PROTO):
    config = copy.deepcopy(config_proto)
    config['anchor_count'] = len(config['anchor_ratios']) * len(config['anchor_scales'])
    config['output_size'] = (
        int(config['image_height'] / config['base_model_stride']),
        int(config['image_width'] / config['base_model_stride'])
    )
    return config
