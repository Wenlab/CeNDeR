# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


from tqdm import tqdm
from multiprocessing import Pool

from common_utils.prints import print_error_message
from preprocessing.inference.auto import preprocess


def auto_preprocess(mode: int, paths, args):
    """

    :param mode: the flag of preprocessing
                0: only return volume after clipping from zrange[0] to zrange[1]

                1: 0 + return maximum projection image
                2: 1 + return the contour of head region
                3: 2 + return its 5 pts of C. elegans Coordinates system

                4: 1 + save maximum projection image
                5: 4 + save the json file of contour of head region
                6: 5 + save 5 pts of C. elegans Coordinates system

    :param args:
    :return:
        0: return volume
        1 & 4: return [volume,
                    projection]
        2 & 5: return [volume,
                projection,
                [convex_head_ctr, is_head_region_warning, convex_head_binary_region, rect_of_roi, tail_ctrs, noise_ctrs]]
        3 & 6: return [volume,
                    projection,
                    [convex_head_ctr, is_head_region_warning, convex_head_binary_region, rect_of_roi, tail_ctrs, noise_ctrs],
                    [mass_of_center, upper_y, lower_y, right_x, left_x]]
    """

    if mode not in (0, 1, 2, 3, 4, 5, 6):
        print_error_message(f"{mode} is not supported!")

    params = [[stack_path, mode, args.zrange, args.load_preprocess_result_root, args.save_preprocess_result_root, args.name_reg] for stack_path in paths]
    with Pool(min(8, len(params))) as p:
        with tqdm(p.imap_unordered(preprocess, params), total = len(params), desc = "Stack loading + S.1 preprocessing") as pbar:
            total_results = {n: r for out in list(pbar) if out for n, r in out[1].items()}

    return total_results
