import os
import json
from datasets import DatasetDict, Dataset
from PIL import Image
#from pathlib import Path

def reformat_json_dataset(path, new_json=False, image_superpath=''):
    new_format = []
    with open(path, encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for anno in data['annotations']:
            label = data['categories'][anno['category_id']-1]['name']
            image_path = data['images'][anno['image_id']-1]['file_name']
            if image_superpath != '':
                image_path = os.path.join(image_superpath, label, image_path)
            new_format.append({'image_path': image_path, 'label': label, 'bbox': anno['bbox']})
    if new_json:
        new_json_path = path[:-5] + "_re.json"
        with open(new_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(new_format, jsonfile)
    return new_format

reformat_json_dataset(os.path.join("datasets", "CN_coin_annotations", "coin_annotations.json"), \
                      new_json=True, image_superpath = os.path.join("datasets", "CN_dataset_obj_detection_04_23", "dataset_obj_detection"))
