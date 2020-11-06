import csv

def annotation_from_label(label):
    image_width = int(label[2])
    image_height = int(label[3])
    if not image_width or not image_height:
        raise ValueError(f'Image {label[1]} has invalid height or width')
    return {
        'image_filename': label[0],
        'image_id': int(label[1]),
        'image_width': image_width,
        'image_height': image_height,
        'object_count': int(label[5]),
        'objects': []
    }

def load_kaggle_annotations(labels_path, boxes_path):
    """
        Loads a Kaggle dataset annotations for head detection formatted like
        the dataset at https://www.kaggle.com/tensor2flow/head-detection-dataset
        Args:
            labels_path, a path to the labels.csv file
            boxes_path, a path to the boxes.csv file
        Returns:
            a list of image_annotation dictionaries
    """
    with open(labels_path, newline='') as labels_file:
        label_reader = csv.reader(labels_file, delimiter=',')
        next(label_reader)
        annotations = [annotation_from_label(label) for label in label_reader]
    annotations.sort(key=lambda annot: annot['image_id']) # make sure annotations are sorted by id
    with open(boxes_path, newline='') as boxes_file:
        box_reader = csv.reader(boxes_file, delimiter=',')
        next(box_reader)
        for box in box_reader:
            index = int(box[1]) - 1
            annotations[index]['objects'].append(
                [int(x) for x in box[2:]]
            )
    return annotations
