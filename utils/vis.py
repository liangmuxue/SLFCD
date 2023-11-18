import time
import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

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

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = [0, 0, 0, 0]
    y[0] = x[0]  # x center
    y[1] = x[1]  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = [0, 0, 0, 0]
    y[0] = x[0]  # x center
    y[1] = x[1]  # y center
    y[2] = x[2] + x[0]  # width
    y[3] = x[3] + x[1]  # height
    return y


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval


def is_keyframe(img_id, interval=10):
    if img_id % interval == 0:
        return True
    else:
        return False


def plti(im, **kwargs):
    plt.imshow(im, interpolation="none", **kwargs)
    plt.show()


def cvshow(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_box_from_list(box_det_list):
    bboxes = []
    for bbox_index, bbox_det_dict in enumerate(box_det_list):
        bboxes.append(bbox_det_dict["bbox"])
    return bboxes


def vis_data(img_data_ori, bboxes=None, box_mode=1, color='blue', not_show=False,thickness=1):
    img_data = np.copy(img_data_ori)
    # return
    if (bboxes is None and not not_show):
        cvshow(img_data)
        # plti(img_data)
        return
    color = find_color_scalar(color)
    for bbox in bboxes:
        if box_mode == 1:
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[0] + bbox[2])
            y_max = int(bbox[1] + bbox[3])
        else:
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])
        cv2.rectangle(img_data,
                      (x_min, y_min),
                      (x_max, y_max),
                      color=color,
                      thickness=thickness)
    if not_show is False:
        # plti(img_data)
        cvshow(img_data)
    return img_data

def visdom_img_data(img_data, title="debug title", cap="debug cap", viz=None):
    if viz is None:
        from visdom import Visdom
        viz = Visdom(env="debug", port=8098)
    raw_img = img_data.transpose(2, 0, 1)[::-1, ...]
    viz.image(
        raw_img,
        opts=dict(title=title, caption=cap)
    )


def visdom_data(img_data_ori, bboxes=None, box_mode=1, color='blue', not_show=False, title=None, caption=None,
                img_single=None, img_single2=None):
    from visdom import Visdom
    img_data = np.copy(img_data_ori)
    color = find_color_scalar(color)
    for bbox in bboxes:
        if box_mode == 1:
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[0] + bbox[2])
            y_max = int(bbox[1] + bbox[3])
        else:
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])
        cv2.rectangle(img_data,
                      (x_min, y_min),
                      (x_max, y_max),
                      color=color,
                      thickness=1)
    if not_show is False:
        if img_single is not None:
            img_single = cv2.resize(img_single, (1920, 1080))
            img_data = np.concatenate((img_data, img_single), axis=1)
        if img_single2 is not None:
            img_single2 = cv2.resize(img_single2, (1920, 1080))
            img_data = np.concatenate((img_data, img_single2), axis=1)
        viz = Visdom(env="kk", port=8098)
        raw_img = img_data.transpose(2, 0, 1)[::-1, ...]
        viz.image(
            raw_img,
            opts=dict(title=title, caption=caption)
        )
        return




def build_pure_keypoints_img1(keypoint, bbox, total_scores,joint_connections=None, color="yellow", r=5):
    """
    只生成关键点示意图，不包含原图
    :param keypoint: 单人关键点
    :param bbox: 单人候选框
    :param joint_connections: 关节点连接线
    :return:
    """

    # bbox = xyxy2xywh(bbox_ori)
    # color_type = find_color_scalar(color)
    # width = int(bbox[2])
    # height = int(bbox[3])
    height, width, _ = bbox.shape
    if height < 456 or width < 456:
        height = 256
        width = 456
    print(height,width)
    box_x = bbox[0]
    box_y = bbox[1]
    # 生成底色为黑色的图，然后把关键点画上去
    img = np.zeros([height, width, 3], np.uint8)
    nkp = []
    for index, p in enumerate(keypoint):
        if index == 14 or index == 15 or index == 16 or index ==17:
            if p[0]!=-1:
                # 需要使用相对坐标
                x = p[0]
                y = p[1]
                color_value = int(255)# * total_scores[index])
                nkp.append([x, y, total_scores[index]])
                img = cv2.circle(img, (int(x), int(y)), r, color=(color_value,color_value,color_value))
    return img, nkp

def save_image(filepath, img_data):
    cv2.imwrite(filepath, img_data)


def crop_candicates(img_data, bbox):
    """
    crop imageto single person image data
    :param img_data:
    :param bbox:
    :return:
    """
    x1 = int(bbox[0])
    if x1 < 0:
        x1 = 0
    y1 = int(bbox[1])
    if y1 < 0:
        y1 = 0
    box_w = int(bbox[2])
    box_h = int(bbox[3])
    crop_img = img_data[y1:y1 + box_h, x1:x1 + box_w, :]
    return crop_img


def show_features(features_arr, name="test"):
    from visdom import Visdom
    viz = Visdom(env="dev")
    l = np.size(features_arr, 1)
    size = np.size(features_arr, 0)
    x_arr = []
    for i in range(0, l):
        x_arr.append(np.arange(0, size))
    x_arr = np.column_stack(tuple(x_arr))
    win = viz.line(
        X=x_arr,
        Y=features_arr,
        opts=dict(markers=False,
                  title=name,
                  xlabel='index',
                  ylabel='feature',
                  fillarea=False),
        # name=name,
        # update='append'
    )


def show_features_single(features, name="test"):
    size = len(features)
    from visdom import Visdom
    viz = Visdom(env="dev")
    viz.line(X=np.arange(0, size), Y=features, opts=dict(markers=False,
                                                         title=name,
                                                         xlabel='index',
                                                         ylabel='feature',
                                                         fillarea=False), )


def visdom_body_debug(image, title, cap):
    from visdom import Visdom
    viz = Visdom(env="debug", port=8098)
    image = cv2.resize(image, (120, 180))
    raw_img = image.transpose(2, 0, 1)[::-1, ...]
    viz.image(
        raw_img,
        opts=dict(title=title, caption=cap)
    )


def visdom_body_debug2(image, title, cap):
    from visdom import Visdom
    image = image.transpose(2, 0, 1)[::-1, ...]
    viz = Visdom(env="debug", port=8098)
    viz.image(
        image,
        opts=dict(title=title, caption=cap)
    )


def repeat_with_color(color_data,target_shape):
    color_data = np.array(color_data)
    color_data_1 = np.expand_dims(color_data,axis=-1).repeat(target_shape[0],axis=-1)
    color_data_2 = np.expand_dims(color_data_1,axis=-1).repeat(target_shape[1],axis=-1)
    color_data_2 = color_data_2.transpose(1,2,0)
    return color_data_2
 

def show_mask_img(color_data,target_shape):
    color_data = np.array(color_data)
    color_data_1 = np.expand_dims(color_data,axis=-1).repeat(target_shape[0],axis=-1)
    color_data_2 = np.expand_dims(color_data_1,axis=-1).repeat(target_shape[1],axis=-1)
    color_data_2 = color_data_2.transpose(1,2,0)
    return color_data_2

def ptl_to_numpy(plt):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)[:, :, :3]
    return image
    
if __name__ == '__main__': 
    color_data = [128,0,0]
    target_shape = [300,600]
    repeat_with_color(color_data,target_shape)
     
       
    