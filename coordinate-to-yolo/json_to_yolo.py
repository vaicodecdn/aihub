import json, os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import pandas as pd
from shutil import copy2

# global variable
DATA_COUNT = 10
TRAN_SPLIT=7
JSON_DIR = './data'
IMG_DIR = './teeth'
YOLO_IMG = './yolo-annotation/images'
YOLO_IMG_DRAW = './yolo-annotation/draws'
YOLO_LABEL = './yolo-annotation/lables'
IMG_EXTENSION = 'png'

# create a directory for yolo dataset
Path(os.path.join(YOLO_IMG_DRAW)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(YOLO_IMG, 'train')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(YOLO_IMG, 'val')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(YOLO_LABEL, 'train')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(YOLO_LABEL, 'val')).mkdir(parents=True, exist_ok=True)

# convert image pixel coordinate to yolo format
def json_to_yolo(filename, draw_box = False, val=False):
    copy2(os.path.join(IMG_DIR, f'{filename}.{IMG_EXTENSION}'), os.path.join(YOLO_IMG,'train' if not val else 'val', f'{filename}.{IMG_EXTENSION}'))
    with open(os.path.join(JSON_DIR, f'{filename}.json'), 'r') as json_file: 
        json_data = json.load(json_file)['annotation']['tooth']
        image = np.array(Image.open(os.path.join(IMG_DIR, f'{filename}.{IMG_EXTENSION}')).convert('L'))
        img_y, img_x = image.shape
        teeth_numbers = list(teeth_number for teeth_number in json_data.keys())
        label, param1, param2, param3, param4 = [], [], [], [], []
        for teeth_number in teeth_numbers:
            teeth_coordinates = json_data[teeth_number]['coordinate']
            x, y = teeth_coordinates[::2], teeth_coordinates[1::2] # odd & even 
            min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
            box_width, box_height = max_x-min_x, max_y-min_y
            center_x, center_y = int((min_x+max_x)/2), int((min_y+max_y)/2)
            label.append(teeth_number)
            param1.append(center_x/img_x)
            param2.append(center_y/img_y)
            param3.append(box_width/img_x)
            param4.append(box_height/img_y)

            # draw the box on the original image
            for index in range(int(len(teeth_coordinates)/2)):
                image[y[index]][x[index]] = 1
            if draw_box:
                image = np.stack((image,)*3, axis=-1) if len(image.shape) <3 else image
                image[center_y][center_x][0] = 255
                image[center_y][center_x][1] = 0
                image[center_y][center_x][2] = 0
                tooth_length = 100 * box_height / img_y
                cv2.rectangle(image, (min_x, min_y),(max_x, max_y), (255,60,190), 1)
                cv2.putText(image, f'{round(tooth_length, 2)}mm', (min_x, min_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (71, 243, 255), 1)
        if draw_box:
            Image.fromarray(image).save(os.path.join(YOLO_IMG_DRAW, f'{filename}.{IMG_EXTENSION}'))
        # end drawing box on original image

        df = pd.DataFrame({'label': label, 'param1': param1, 'param2': param2, 'param3': param3, 'param4': param4})
        df.to_csv(os.path.join(YOLO_LABEL,'train' if not val else 'val', f'{filename}.csv'), index=False, header=False, sep=' ')


if '__main__' == __name__ :
    for index, filename in enumerate(os.listdir(JSON_DIR)[:DATA_COUNT]):
        format_filename = os.path.splitext(filename)[0]
        if index < TRAN_SPLIT:
            json_to_yolo(filename=format_filename, draw_box=True)
        else:
            json_to_yolo(filename=format_filename, draw_box=True, val=True)