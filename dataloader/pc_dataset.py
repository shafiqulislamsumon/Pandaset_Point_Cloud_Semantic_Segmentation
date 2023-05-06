import os
import yaml
import numpy as np
import json
import gzip
import pickle
from PIL import Image
from torch.utils import data
from pathlib import Path

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


@register_dataset
class Pandaset(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.im_idx = []

        self.camera = 'camera/front_camera'
        self.annotation = 'annotations/semseg'
        self.lidar = 'lidar'
        self.points_cols = ['x', 'y', 'z']
        self.depth_col = ['i']
        self.class_col = ['class']
        self.mapping = 'classes.json'

        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        for i_folder in split:
            self.im_idx += self.absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(3), self.lidar]), num_vote)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def absoluteFilePaths(self, directory, num_vote):
        for dirpath, _, filenames in os.walk(directory):
            filenames = [file for file in filenames if not (file.endswith('.DS_Store') or file.endswith('.json'))]
            filenames.sort()
            for f in filenames:
                for _ in range(num_vote):
                    yield os.path.abspath(os.path.join(dirpath, f))

    def __getitem__(self, index):
        with gzip.open(self.im_idx[index], "rb") as fin:
            lidar_data = pickle.load(fin)

        # Lidar dataframe columns = ['x', 'y', 'z', 'i', 't', 'd']
        data_len = len(lidar_data)
        points = lidar_data[self.points_cols].to_numpy()
        depth = lidar_data[self.depth_col].to_numpy()

        image_file = self.im_idx[index].replace(self.lidar, self.camera).replace('.pkl.gz', '.jpg')
        image = Image.open(image_file)

        annotation_file = self.im_idx[index].replace(self.lidar, self.annotation)
        dir_name = os.path.dirname(annotation_file)
        mapping_file = os.path.join(dir_name, self.mapping)

        with open(mapping_file) as user_file:
            # currently no usage
            mapping_data = json.load(user_file)

        with gzip.open(annotation_file, "rb") as fin:
            annotated_data = pickle.load(fin)
            annotated_data = annotated_data[self.class_col].to_numpy()
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
    
        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = annotated_data
        data_dict['signal'] = depth
        data_dict['origin_len'] = data_len
        data_dict['img'] = image

        return data_dict, self.im_idx[index]


def get_Pandaset_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        pandaset_yaml = yaml.safe_load(stream)

    Pandaset_label_name = dict()
    for i in sorted(list(pandaset_yaml['learning_map'].keys()))[::-1]:
        Pandaset_label_name[pandaset_yaml['learning_map'][i]] = pandaset_yaml['labels'][pandaset_yaml['learning_map'][i]]

    #print("##### Pandaset_label_name : ", Pandaset_label_name)
    return Pandaset_label_name
