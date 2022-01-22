# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import json
from typing import Dict, List, Tuple

from common_utils.prints import print_error_message

CartesianCoordLike = Tuple[int, int]


def convert_head_ctr_to_dict(head_ctr, is_head_region_warning: bool) -> Dict:
    """
    Attention: the value 0 of "group_id" means the region must be a head, and 1 may be a head.
    :param head_ctr: (n, 1, 2). Contour of head region from cv2.findContour
    :param is_head_region_warning: Boolean type
    :return: an element in value of "shape" key of labelme annotation json file
    """

    if isinstance(head_ctr, List):
        head_ctr = head_ctr[0]
    ctn = {
        "label"     : "head",
        "group_id"  : 1 if is_head_region_warning else 0,
        "shape_type": "polygon",
        "flags"     : dict(),
        "points"    : [[float(p[0][0]), float(p[0][1])] for p in head_ctr]
    }

    return ctn


def convert_tail_ctrs_to_dict(tail_ctrs) -> List[Dict]:
    """
    the number in "group_id" means the relative index of tail contours.

    :param tail_ctrs: [(n, 1, 2) x N], N is the number of tail contours. Contours of tail regions from cv2.findContour
    :return: a list of element in value of "shape" key of labelme annotation json file
    """

    ctns = [{"label" : "tail", "group_id": i, "shape_type": "polygon", "flags": dict(),
             "points": [[float(p[0][0]), float(p[0][1])] for p in ctr]} for i, ctr in enumerate(tail_ctrs)]

    return ctns


def convert_pts_to_dict(pts: List[CartesianCoordLike]):
    """

    :param pts: [mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x]
    :return:  five points in value of "shape" key of labelme annotation json file
    """

    labels = ["mass_of_center", "anterior_y", "posterior_y", "ventral_x", "dorsal_x"]
    ctns = [{"label" : l, "shape_type": "point", "flags": dict(), "group_id": None,
             "points": [[float(p[0]), float(p[1])]]} for l, p in zip(labels, pts)]

    return ctns


def complete_anno(ctns: List[Dict], image_path: str, image_height: int, image_width: int) -> Dict:
    """

    :param ctns:
    :param image_height:
    :param image_width:
    :param image_path:
    :return:
    """

    ctn = {
        "version"    : "4.5.7",
        "flags"      : dict(),
        "imagePath"  : image_path,
        "imageData"  : None,
        "imageHeight": image_height,
        "imageWidth" : image_width,
        "shapes"     : ctns,
    }

    return ctn


def save_dict_as_json(json_path: str, content: Dict):
    content = json.dumps(content, indent = 2)
    with open(json_path, "w", encoding = "utf-8") as json_file:
        json_file.write(content)


def save_anno_as_json(mode: int, json_path: str, **kwargs):
    if mode in (5, 6):
        content = [convert_head_ctr_to_dict(kwargs['head_ctr'], kwargs['is_head_region_warning'])]
        content = content if len(kwargs['tail_ctrs']) == 0 else content + convert_tail_ctrs_to_dict(kwargs['tail_ctrs'])
        content = content if mode == 5 else content + convert_pts_to_dict(kwargs['pts'])
    else:
        print_error_message(f"Mode {mode} is not supported !")
    content = complete_anno(ctns = content, image_path = kwargs['image_path'], image_height = kwargs['image_height'], image_width = kwargs['image_width'])
    save_dict_as_json(json_path, content)
