'''
 detection_visualizer.py
 Visualizer for Candidate Detection
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on June 18th, 2018
'''
import matplotlib

import matplotlib.pyplot as plt
from random import random as rand

import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from keypoint_visualizer import add_joint_connections_with_lines, show_poses_from_python_data
from utils_io_file import is_image
from utils_io_folder import create_folder
from utils_json import read_json_from_file
from utils_log import LogUtil

bbox_thresh = 0.4
reduce_height = 45

# set up class names for COCO
num_classes = 81  # 80 classes + background class
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def show_boxes_from_python_data(img, dets, classes, output_img_path, scale=1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(img)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    plt.savefig(output_img_path)
    return img


def find_color_scalar(color_string):
    color_dict = {
        'darkslategray': (79, 47, 79),
        'orangered3': (0, 205, 55),
        'purple': (255, 0, 255),
        'yellow': (0, 255, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'skyblue': (235, 206, 135),
        'navyblue': (128, 0, 0),
        'azure': (255, 255, 240),
        'slate': (255, 0, 127),
        'chocolate': (30, 105, 210),
        'olive': (112, 255, 202),
        'orange': (0, 140, 255),
        'orchid': (255, 102, 224),
        'floralwhite': (240, 250, 250),
        'mediumblue': (205, 0, 0)
    }
    color_scalar = color_dict[color_string]
    return color_scalar


def draw_box_single(img, bbox, color):
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    cv2.rectangle(img, pt1, pt2, color, 3)
    return img


def draw_bbox_withname(img, bbox, name, track_id=-1, img_id=-1):
    if track_id == -1:
        # this is for objects
        color = find_color_scalar('floralwhite')
        text_color = find_color_scalar('mediumblue')
    else:
        color_list = ['purple', 'yellow', 'blue', 'green', 'red', 'skyblue', 'navyblue', 'azure', 'slate', 'chocolate',
                      'olive', 'orange', 'orchid']
        # color_name = color_list[track_id % 13]
        color = find_color_scalar('orange')
        text_color = find_color_scalar('darkslategray')

    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    cv2.rectangle(img, pt1, pt2, color, 3)

    img = add_text(img, name, bbox[0], bbox[1] - reduce_height, color=text_color)
    return img


def draw_text(img, text_list):
    if len(text_list) == 0:
        return img
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for text_dict in text_list:
        box = text_dict["box"]
        color = text_dict["color"]
        text = text_dict["text"]
        left = box[0]
        top = box[1] - reduce_height
        if left < 10 or top < 10:
            return img
        color_array = np.array(color).astype(int)
        color = (color_array[1], color_array[2], color_array[0])
        font = ImageFont.truetype('fonts/msyh.ttf', 40)
        position = (int(left), int(top))
        if not isinstance(text, np.unicode):
            text = text.decode('utf8')
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, text, font=font, fill=color)
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV


def draw_text_opencv(img, text_list, letter_dict=None):
    if len(text_list) == 0:
        return img
    for text_dict in text_list:
        box = text_dict["box"]
        color = text_dict["color"]
        text = text_dict["text"]
        left = box[0]
        top = box[1] - reduce_height
        if left < 10 or top < 10:
            return img
        color_array = np.array(color).astype(float)
        position = (int(left), int(top))
        img = append_letter_img(img, position, text, letter_dict=letter_dict)
    return img


def append_letter_img(img, position, letter, letter_dict=None):
    logger = LogUtil()
    # logger.track_log("track_letter", "letter in append:{}".format(letter))
    (left, top) = position
    letter2 = None
    if "," in letter:
        letters = letter.split(",")
        letter = letters[0]
        letter2 = letters[1]
    if letter in letter_dict:
        s_img = letter_dict[letter]
        img[top:top + s_img.shape[0], left:left + s_img.shape[1]] = s_img
    if letter2 is not None:
        if letter2 in letter_dict:
            s_img2 = letter_dict[letter2]
            img[top:top + s_img2.shape[0],
            left + s_img.shape[1] + 10:left + s_img.shape[1] + 10 + s_img2.shape[1]] = s_img2
    return img


def add_text(img, text, left, top, color=None, font_size=12, thickness=1, lineType=None):
    if left < 10 or top < 10:
        return img

    color_array = np.array(color).astype(int)
    color = (color_array[1], color_array[2], color_array[0])
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('fonts/msyh.ttf', 40)
    position = (int(left), int(top))
    if not isinstance(text, np.unicode):
        text = text.decode('utf8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=color)
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV


def draw_bbox(img, bbox, score, class_id, track_id=-1, img_id=-1):
    if track_id == -1:
        color = (255 * rand(), 255 * rand(), 255 * rand())
    else:
        color_list = ['purple', 'yellow', 'blue', 'green', 'red', 'skyblue', 'navyblue', 'azure', 'slate', 'chocolate',
                      'olive', 'orange', 'orchid']
        color_name = color_list[track_id % 13]
        color = find_color_scalar(color_name)

    if img_id % 10 == 0:
        color = find_color_scalar('red')
    elif img_id != -1:
        color = find_color_scalar('blue')

    cv2.rectangle(img,
                  (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  color=color,
                  thickness=3)

    cls_name = classes[class_id]
    if class_id > 0:
        cls_name = cls_name + "_" + str(track_id)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if track_id == -1:
        cv2.putText(img,
                    # '{:s} {:.2f}'.format(cls_name, score),
                    '{:s}'.format(cls_name),
                    (bbox[0], bbox[1] - 5),
                    font,
                    fontScale=0.8,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)
    else:
        cv2.putText(img,
                    # '{:s} {:.2f}'.format("ID:"+str(track_id), score),
                    '{:s}'.format("ID:" + cls_name),
                    (bbox[0], bbox[1] - 5),
                    font,
                    fontScale=0.8,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)
    return img


def show_boxes_from_standard_json(json_file_path, classes, img_folder_path=None, output_folder_path=None, track_id=-1):
    dets = read_json_from_file(json_file_path)

    for det in dets:
        python_data = det

        if img_folder_path is None:
            img_path = os.path.join(python_data["image"]["folder"], python_data["image"]["name"])
        else:
            img_path = os.path.join(img_folder_path, python_data["image"]["name"])
        if is_image(img_path):    img = cv2.imread(img_path)

        candidates = python_data["candidates"]
        for candidate in candidates:
            bbox = np.array(candidate["det_bbox"]).astype(int)
            score = candidate["det_score"]
            if score >= bbox_thresh:
                img = draw_bbox(img, bbox, score, classes, track_id=track_id)

        if output_folder_path is not None:
            create_folder(output_folder_path)
            img_output_path = os.path.join(output_folder_path, python_data["image"]["name"])
            cv2.imwrite(img_output_path, img)
    return True


def visualize_img(img, candidates, img_id, flag_track=True, flag_showkeypoints=False, letter_dict=None):
    obj_color = find_color_scalar('floralwhite')
    obj_text_color = find_color_scalar('mediumblue')
    humman_color = find_color_scalar('orange')
    humman_text_color = find_color_scalar('darkslategray')

    text_list = []

    for candidate in candidates:
        bbox = np.array(candidate["bbox"]).astype(int)
        if "class_id" in candidate:
            class_id = candidate["class_id"]
            cls_name = classes[class_id]
        else:
            # reid matcher reguler
            if 'human_name' in candidate:
                cls_name = candidate["human_name"]
            else:
                cls_name = "stranger"
        # optional: show the bounding boxes
        track_id = candidate["track_id"]
        if track_id is None:
            img = draw_box_single(img, bbox, obj_color)
        elif "class_id" in candidate:
            # for object ,only show box
            img = draw_box_single(img, bbox, obj_color)
            text_list.append({"box": bbox, "text": cls_name, "color": obj_text_color})
        else:
            act_name = candidate["act_name"]
            if len(act_name) > 0:
                total_text = cls_name + "," + act_name
            else:
                total_text = cls_name
            img = draw_box_single(img, bbox, humman_color)
            text_list.append({"box": bbox, "text": total_text, "color": humman_text_color, "human": 1})
        img = draw_text_opencv(img, text_list, letter_dict=letter_dict)
        # img = draw_text(img, text_list)

        # only show bbox
        if flag_showkeypoints is False:
            continue

        if "joints_connection" in candidate:
            joints_connection = candidate["joints_connection"]
        else:
            joints_connection = []

        track_id = candidate["track_id"]
        img = add_joint_connections_with_lines(img, joints_connection)

    return img
