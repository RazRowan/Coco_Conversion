import json
import os
from os import listdir
import cv2
import time

image_list = []
new_annotation_list = []


# Open the original JSON file, store it as a variable, and close the file
file = open('./data/annotations.json')
original_data = json.load(file)
file.close()

# Create the new JSON file with a framework (info can be changed here if you want)
new_coco_data = \
    {
        "info": {
            "year": "2022",
            "version": "1",
            "description": "Exported from roboflow.ai",
            "contributor": "",
            "url": "https://app.roboflow.ai/datasets/blueberries_moore_drone_6-16-2021/1",
            "date_created": "2022-01-05T20:45:25+00:00"
        },
        "licenses": [
            {
                "id": 1,
                "url": "",
                "name": "Unknown"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "berries",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "blue",
                "supercategory": "berries"
            },
            {
                "id": 2,
                "name": "green",
                "supercategory": "berries"
            }
        ],
        "images": [],
        "annotations": []
    }

def convert():
    global image_list, new_annotation_list

    add_images()
    add_annotations()
    write_to_json(image_list, new_annotation_list)

def add_images():
    global image_list, file

    # get the path/directory
    folder_dir = "./data/"

    index = 0
    for file in os.listdir(folder_dir):
        # check if the image ends with png
        if (file.endswith(".png") or file.endswith(".jpg")):
            image = cv2.imread(folder_dir + file)
            height, width = image.shape[:2]
            date_captured = str(os.path.getctime(folder_dir + file))
            write_new_image(0, 1, file, height, width, date_captured, image_list)
        index += 1




def add_annotations():
    old_annotations = original_data[0]['boxes']
    for i in range(len(old_annotations)):
        old_index = i
        old_class = old_annotations[str(i)]['class']
        old_xmax = int(old_annotations[str(i)]['xmax'])
        old_xmin = int(old_annotations[str(i)]['xmin'])
        old_ymax = int(old_annotations[str(i)]['ymax'])
        old_ymin = int(old_annotations[str(i)]['ymin'])

        new_width = old_xmax - old_xmin
        new_height = old_ymax - old_ymin

        new_bbox = [
            old_xmin,
            old_ymin,
            new_width,
            new_height
        ]

        new_area = new_width * new_height

        write_new_annotation(old_index, 0, old_class, new_bbox, new_area, [], 0, new_annotation_list)




# Method to write new image in coco format
def write_new_image(id, license, file_name, height, width, date_captured, image_list):
    new_image = {
        "id": id,
        "license": license,
        "file_name": file_name,
        "height": height,
        "width": width,
        "date_captured": date_captured
    }

    image_list.append(new_image)

# Method to write new annotation in coco format
def write_new_annotation(id, image_id, category_id, bbox, area, segmentation, iscrowd, new_annotation_list):
    new_annotation = {
        "id" : id,
        "image_id" : image_id,
        "category_id" : category_id,
        "bbox" : bbox,
        "area" : area,
        "segmentation" : segmentation,
        "iscrowd" : iscrowd
    }

    new_annotation_list.append(new_annotation)

# Separate method for appending to image_list, and writing
def write_to_json(image_list, new_annotation_list):
    for i in range(len(image_list)):
        new_coco_data['images'].append(image_list[i])

    for i in range(len(new_annotation_list)):
        new_coco_data['annotations'].append(new_annotation_list[i])

    # Save everything to the new JSON file
    with open('./result/_new_annotations.coco.json', 'w') as f:
        f.write(json.dumps(new_coco_data, indent=4))



convert()