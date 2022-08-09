# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_pt_cloud_mip(pts, image_size):
    pts = np.array(pts, dtype = np.int32)
    assert pts.shape[1] in (2, 3)
    mip = np.zeros(image_size, dtype = np.uint8)
    for pt in pts:
        cv2.circle(mip, tuple(pt), 2, 1, 3)
    return mip


def cvt_res(output, preprc_res):
    """
    Convert det and rec output from a cropped volume into a raw volume.
    Parameters
    ----------
    output
    preprc_res

    Returns
    -------

    """

    ox, oy, oz = [int(x) for x in preprc_res[2][3][:2]] + [0]
    new_output = {int(key): [[i[0] + ox, i[1] + oy, i[2] + ox, i[3] + oy, i[4] + oz] for i in output[key]] for key in sorted(output.keys())}

    return new_output


def draw_volume_result(output, preprc_res, save_fig_root, name, verbose: bool = False, others_class_start_id: int = 2000, shift: int = 1):
    volume, head_binary_region, rr = preprc_res[0], preprc_res[2][2], preprc_res[2][3]
    crv = (volume[rr[1]: rr[1] + rr[3], rr[0]: rr[0] + rr[2]]).astype(np.float32)
    crv = (crv - crv.mean()) / crv.std()
    crv = crv * head_binary_region[rr[1]: rr[1] + rr[3], rr[0]: rr[0] + rr[2], np.newaxis]

    plt.figure(figsize = (25, 20))
    # fig.tight_layout()
    for idx in range(crv.shape[-1]):
        plt.subplot(5, 4, idx + 1)
        img = crv[..., idx]
        plt.imshow(img, cmap = "inferno", vmin = crv.min(), vmax = 15)
        # draw box and id
        for _id, neuron in output.items():
            for reg in neuron:
                if reg[-1] == idx:
                    plt.gca().add_patch(plt.Rectangle((reg[0], reg[1]), reg[2] - reg[0], reg[3] - reg[1], linewidth = 1, edgecolor = 'r', facecolor = "None"))
                    plt.text((reg[0] + reg[2]) * 0.5, (reg[1] + reg[3]) * 0.5, s = str(_id + shift) if _id >= others_class_start_id else "-1",
                             verticalalignment = 'center', horizontalalignment = 'center',
                             fontdict = {"color": "blue", "fontsize": 4, "weight": "bold"})

        plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
        plt.title(f"frame {idx + shift}")
        plt.axis('off')
        plt.margins(0, 0)
    plt.savefig(f"{save_fig_root}/{name}.pdf")
    if verbose:
        plt.show()
