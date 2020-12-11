import csv
import os
import xml.etree.ElementTree as ET # phone home

def annotation_from_kaggle_label(label):
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
            boxes_path,  a path to the boxes.csv file
        Returns:
            a list of image_annotation dictionaries
    """
    with open(labels_path, newline='') as labels_file:
        label_reader = csv.reader(labels_file, delimiter=',')
        next(label_reader)
        annotations = [annotation_from_kaggle_label(label) for label in label_reader]
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

def _map_object_node_to_bbox(object_node):
    bbox_node = object_node.find('bndbox')
    xmin = int(bbox_node.find('xmin').text)
    xmax = int(bbox_node.find('xmax').text)
    ymin = int(bbox_node.find('ymin').text)
    ymax = int(bbox_node.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

def annotation_from_xml_file(basepath, filename):
    filepath = os.path.join(basepath, filename)
    xml_tree = ET.parse(filepath)
    root = xml_tree.getroot()
    filename = f'{os.path.splitext(filename)[0]}.jpeg'
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    gt_boxes = [_map_object_node_to_bbox(object_node) for object_node in root.findall('object')]
    return {
        'image_filename': filename,
        'image_width': width,
        'image_height': height,
        'object_count': len(gt_boxes),
        'objects': gt_boxes
    }

def load_xml_annotations(annotations_folder):
    """
        Loads the scut head dataset for head detection
        Args:
            anotations_folder, the path to the folder containing annotations to load
        Returns:
            a list of image annotation dictionaries
    """
    return [annotation_from_xml_file(annotations_folder, filename) for filename in os.listdir(annotations_folder)]

def _apply_basepath_to_annotation(annotation, basepath):
    return  {
        'image_filename': os.path.join(basepath, annotation['image_filename']),
        'image_width': annotation['image_width'],
        'image_height': annotation['image_height'],
        'object_count': annotation['object_count'],
        'objects': annotation['objects']
    }

def aggregate_annotations(annotation_lists, basepaths):
    """
        Aggregates the annotations into a single list and add the corresponding
        basepath to each filename
        Args:
            annotation_lists, the datasets to aggregate
            basepaths,        the basepath for each dataset
        Returns:
            the aggregated annotations
    """
    aggregated_annotations = []
    for annotations, basepath in zip(annotation_lists, basepaths):
        aggregated_annotations += [_apply_basepath_to_annotation(annotation, basepath) for annotation in annotations]
    return aggregated_annotations
