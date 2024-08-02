import copy
import sys
import traceback

import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as tm
from torchvision import transforms
import numpy as np
import cv2
import skimage

COLORS = {
    'BLACK': 30, "RED": 31, "GREEN": 32, "BLUE": 34, "WHITE": 37, \
    "YELLOW": 33, "YANGHONG": 35, "CYAN": 36
}

COLOR_CONSTRUCT = "\033[0;{}m{}\033[0m"

last_color = 31
WARNING = 2
INFO = 1
FLOW = 0
TIPS = 4
ERROR = 3
HIGH_LIGHT = 5

LEVEL_INFO = {
    WARNING: "WARNING",
    INFO: "INFO",
    FLOW: "FLOW",
    TIPS: "TIPS",
    ERROR: "ERROR",
    HIGH_LIGHT: 'HIGH_LIGHT'
}

LEVEL_COLOR = {
    WARNING: "YELLOW",
    INFO: "WHITE",
    FLOW: "BLUE",
    HIGH_LIGHT: "CYAN",
    TIPS: "GREEN",
    ERROR: "RED"
}

# get file by index
gfi = lambda img, ind: copy.deepcopy(img[ind[0]:ind[1], ind[2]:ind[3]])


def color_str(*args, color=COLORS['BLACK']):
    if len(args) == 1:
        return COLOR_CONSTRUCT.format(color, args[0])
    return [COLOR_CONSTRUCT.format(color, a) for a in args]


def err(*args, sep=' ', exit=True, show_traceback_line=5):
    p = sep.join([str(a) for a in args])
    file = __file__.split("/")[-1]
    funcName = sys._getframe().f_back.f_code.co_name  # 获取调用函数名
    lineNumber = sys._getframe().f_back.f_lineno

    info_str, funcName, lineNumber, p = \
        color_str(LEVEL_INFO[ERROR], funcName, lineNumber, p, color=COLORS[LEVEL_COLOR[ERROR]])

    p = "[{}][{}:{}] {}". \
        format(info_str, funcName, lineNumber, p)
    print(p)
    fs = traceback.format_stack()
    for line in fs[0 - show_traceback_line:len(fs)]:
        p = color_str(line.strip(), color=COLORS[LEVEL_COLOR[ERROR]])
        print(p)

    if exit:
        sys.exit()


def warn(*args, sep=' '):
    p = sep.join([str(a) for a in args])
    file = __file__.split("/")[-1]
    funcName = sys._getframe().f_back.f_code.co_name  # 获取调用函数名
    lineNumber = sys._getframe().f_back.f_lineno

    info_str, file, funcName, lineNumber, p = \
        color_str(LEVEL_INFO[WARNING], file, funcName, lineNumber, p, color=COLORS[LEVEL_COLOR[WARNING]])

    p = "[{}][{}][{}:{}] {}". \
        format(info_str, file, funcName, lineNumber, p)
    print(p)


def remove_small_hole(mask, h_size=10):
    """remove the small hole

    Args:
        mask (_type_): a binary mask, can be 0-1 or 0-255
        h_size (int, optional): min_size of the hole

    Returns:
        mask
    """
    value = np.unique(mask)
    if len(value) > 2:
        err(f"Input mask should be a binary, but get value:({value})")
    pre_mask_rever = mask == 0
    pre_mask_rever = skimage.morphology.remove_small_objects(pre_mask_rever, \
                                                             min_size=h_size)
    mask[pre_mask_rever <= 0] = 1
    return mask


def ostu_seg_tissue(img, remove_hole: bool = True, min_size: int = 4000, mod: str = 'cutoff'):
    """Segmentation tissue of WSI with ostu.

    Args:
        img (_type_): Input img. Must in 10X resolution.
        remove_hole (bool, optional): remove small hole or not
        min_size (int, optional): min_size of hole

    Returns:
        _type_: _description_
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise using a Gaussian filter
    gray = cv2.GaussianBlur(gray, (35, 35), 0)
    # Otsu thresholding and mask generation
    if mod == "cutoff":
        thresh_otsu = gray < 234
    elif mod == "ostu":
        ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if remove_hole:
        thresh_otsu = remove_small_hole(thresh_otsu, min_size)
        thresh_otsu = thresh_otsu != 0
        thresh_otsu = remove_small_hole(thresh_otsu, min_size)
    else:
        thresh_otsu = thresh_otsu != 0
    return thresh_otsu


def gen_patches_index(ori_size, *, img_size=224, stride=224, keep_last_size=False):
    """
        这个函数用来按照输入的size和patch大小，生成每个patch所在原始的size上的位置

        keep_last_size：表示当size不能整除patch的size的时候，最后一个patch要不要保持输入的img_size

        返回：
            一个np数组，每个成员表示当前patch所在的x和y的起点和终点如：
                [[x_begin,x_end,y_begin,y_end],...]
    """
    height, width = ori_size[:2]
    index = []
    if height < img_size or width < img_size:
        warn("input size is ({} {}), small than img_size:{}".format(height, width, img_size))
        return index

    for h in range(0, height + 1, stride):
        xe = h + img_size
        if h + img_size > height:
            xe = height
            h = xe - img_size if keep_last_size else h

        for w in range(0, width + 1, stride):
            ye = w + img_size
            if w + img_size > width:
                ye = width
                w = ye - img_size if keep_last_size else w
            index.append(np.array([h, xe, w, ye]))

            if ye == width:
                break
        if xe == height:
            break
    return index


class MyZip(object):
    def __init__(self, *args, batch=1):
        self.value_len = len(args[0])
        for num, value in enumerate(args):
            if not hasattr(value, "__iter__") and not hasattr(value, "__getitem__"):
                err("The {} value has not attr {} and {}".format(num, "__iter__", "__getitem__"))
                sys.exit()
            elif len(value) != self.value_len:
                err("The zeros arg len ({}) donet equal to the {} arg len ({})". \
                    format(self.value_len, num, len(value)))
                sys.exit()
        self.value = args
        self.batch = batch

    def __len__(self):
        if self.value_len % self.batch > 0:
            return self.value_len // self.batch + 1
        else:
            return self.value_len // self.batch

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()
        start = index * self.batch
        end = start + self.batch if start + self.batch <= self.value_len else self.value_len
        ret = []
        for i in self.value:
            ret.append(i[start:end])

        return ret[:]


def gen_mask(model, img, file_name, device):
    img_size = 224
    stride = 56
    batchsize = 512
    indx_list = gen_patches_index(img.shape[0:2], img_size=img_size, stride=stride, keep_last_size=True)
    indx_zip = MyZip(indx_list, batch=batchsize)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    transform = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        for num, ind in tqdm(enumerate(indx_zip), desc=f'seg img {file_name}', total=len(indx_zip)):
            inds = ind[0]
            input_img_list = []
            for _ind in inds:
                input_img_list.append(transform(gfi(img, _ind)))
            input_img_list = torch.stack(input_img_list).to(device)
            pred = model(input_img_list).cpu().detach().numpy()
            for num, _ind in enumerate(inds):
                if pred[num][0] > pred[num][1]:
                    mask[_ind[0]:_ind[1], _ind[2]:_ind[3]] += 0
                else:
                    mask[_ind[0]:_ind[1], _ind[2]:_ind[3]] += 1
    mask = np.array((mask > 4) * 255, dtype=np.uint8)
    return mask


def piplineone(model, img, file_name, device):
    img = np.pad(img, ((224, 224), (224, 224), (0, 0)), mode='reflect')
    mask = gen_mask(model, img, file_name, device)
    ostu_mask = ostu_seg_tissue(img)
    mask = mask * (ostu_mask > 0)
    mask = mask[224:mask.shape[0] - 224, 224:mask.shape[1] - 224]
    return mask


def load_Seg_model(seg_path, device):
    model = tm.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(seg_path, map_location=device))  # 将与训练权重载入模型
    model = model.to(device)
    model.eval()
    return model
