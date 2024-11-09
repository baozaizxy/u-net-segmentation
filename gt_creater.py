import os
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import re

with open('./seaturtleid2022/turtles-data/data/annotations.json', 'r', encoding='utf8') as file:
    data = json.load(file)

# initialise COCO API for annotations
coco = COCO('./seaturtleid2022/turtles-data/data/annotations.json')
root = './seaturtleid2022/turtles-data/data/'
GT_path = './seaturtleid2022/turtles-data/data/images_GT'

os.makedirs(GT_path, exist_ok=True)

image_ids = coco.getImgIds()
num = 0

for image_id in image_ids:
    img = coco.loadImgs(image_id)[0]
    image_path = os.path.join(root, img['file_name'])
    ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros((img['height'], img['width']), dtype=np.uint8)

    for ann in anns:
        mask += coco.annToMask(ann) * ann['category_id'] * 100

    mask_img = Image.fromarray(mask)

    def format_filename(filename):
        simple_filename = re.sub(r'images/', '', filename)
        mask_filename = re.sub(r'\.(jpg|jpeg|JPG|JPEG)$', '_mask.png', simple_filename, flags=re.IGNORECASE)
        return mask_filename

    mask_filename = format_filename(img['file_name'])
    mask_path = os.path.join(GT_path, mask_filename)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    mask_img.save(mask_path)

    num += 1
    print(f'Mask for image {image_id} saved to {mask_path}, it‘s the {num} img')