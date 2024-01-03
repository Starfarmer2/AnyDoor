import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import os
import pandas as pd
import shutil

from pycocotools import mask as coco_mask


DATASET_PATH = '../FashionInpaintingDataGeneration/formatted_dataset'
OUTPUT_PATH = './path/TryOn/VitonHD/test'
DATAPOINT_PATH_WIDTH = 10

os.makedirs(OUTPUT_PATH, exist_ok=True)
print(os.path.join(OUTPUT_PATH, 'cloth'))
os.makedirs(os.path.join(OUTPUT_PATH, 'cloth'), exist_ok=True) #reference
os.makedirs(os.path.join(OUTPUT_PATH, 'cloth-mask'), exist_ok=True) #reference mask
os.makedirs(os.path.join(OUTPUT_PATH, 'image'), exist_ok=True) #target
os.makedirs(os.path.join(OUTPUT_PATH, 'image-parse-v3'), exist_ok=True) #target mask

def img_to_square(image):
    width, height = image.size
    delta_w = height - width
    delta_h = width - height
    padding = (max(delta_w, 0) // 2, max(delta_h, 0) // 2, max(delta_w, 0) // 2, max(delta_h, 0) // 2)

    padded_image = ImageOps.expand(image, padding, fill='black')
    return padded_image


datapoint_folders = sorted(os.listdir(DATASET_PATH))
for datapoint_folder in datapoint_folders:
    # #reference
    # reference_dir = os.path.join(DATASET_PATH, datapoint_folder,'reference.jpg')
    # out_reference_dir = os.path.join(OUTPUT_PATH, 'cloth', f'{datapoint_folder}.jpg')
    # shutil.copy(reference_dir, out_reference_dir)

    # #reference mask
    # reference_mask_dir = os.path.join(DATASET_PATH, datapoint_folder,'reference_mask.jpg')
    # out_reference_mask_dir = os.path.join(OUTPUT_PATH, 'cloth-mask', f'{datapoint_folder}.jpg')
    # shutil.copy(reference_mask_dir, out_reference_mask_dir)

    # #target
    # target_dir = os.path.join(DATASET_PATH, datapoint_folder,'target.jpg')
    # out_target_dir = os.path.join(OUTPUT_PATH, 'image', f'{datapoint_folder}.jpg')
    # shutil.copy(target_dir, out_target_dir)

    # #target mask
    # target_mask_dir = os.path.join(DATASET_PATH, datapoint_folder,'target_mask.png')
    # out_target_mask_dir = os.path.join(OUTPUT_PATH, 'image-parse-v3', f'{datapoint_folder}.png')
    # shutil.copy(target_mask_dir, out_target_mask_dir)


    #reference
    reference_dir = os.path.join(DATASET_PATH, datapoint_folder,'reference.jpg')
    out_reference_dir = os.path.join(OUTPUT_PATH, 'cloth', f'{datapoint_folder}.jpg')
    image = img_to_square(Image.open(reference_dir))
    image.save(out_reference_dir)

    #reference mask
    reference_mask_dir = os.path.join(DATASET_PATH, datapoint_folder,'reference_mask.jpg')
    out_reference_mask_dir = os.path.join(OUTPUT_PATH, 'cloth-mask', f'{datapoint_folder}.jpg')
    image = img_to_square(Image.open(reference_mask_dir))
    image.save(out_reference_mask_dir)

    #target
    target_dir = os.path.join(DATASET_PATH, datapoint_folder,'target.jpg')
    out_target_dir = os.path.join(OUTPUT_PATH, 'image', f'{datapoint_folder}.jpg')
    image = img_to_square(Image.open(target_dir))
    image.save(out_target_dir)

    #target mask
    target_mask_dir = os.path.join(DATASET_PATH, datapoint_folder,'target_mask.png')
    out_target_mask_dir = os.path.join(OUTPUT_PATH, 'image-parse-v3', f'{datapoint_folder}.png')
    image = img_to_square(Image.open(target_mask_dir))
    image.save(out_target_mask_dir)




