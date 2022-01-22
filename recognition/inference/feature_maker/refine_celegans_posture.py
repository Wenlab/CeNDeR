# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import cmath
from typing import List


def refine_celegans_posture(neurons: List[List],
                            ccords: List
                            ):
    """Correct posture of C.elegans

    :param neurons: value is Cartesian Coordinate System (x, y, z)
    :param ccords: 5 pts(x, y) [mass_of_center, anterior_y, posterior_y, dorsal_x, ventral_x]

    :return:
    """

    neurons = neurons.copy()
    mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x = ccords.copy()

    # 1 Zero-centered
    neurons = [[[pt[0] - mass_of_center[0], pt[1] - mass_of_center[1], pt[2]] for pt in pts] for pts in neurons]
    anterior_y = [a - b for a, b in zip(anterior_y, mass_of_center)]
    posterior_y = [a - b for a, b in zip(posterior_y, mass_of_center)]
    ventral_x = [a - b for a, b in zip(ventral_x, mass_of_center)]

    # 2 Transfer tail-head direction into y-axis positive direction (python layout: positive y-axis
    # 2.1 Coordinate transformation: Cartesian -> Polar: (x, y, z) -> (rho, phi, z), (x, y) -> (rho, phi)
    neurons = [[[*cmath.polar(complex(pt[0], pt[1])), pt[2]] for pt in pts] for pts in neurons]
    posterior_y = [*cmath.polar(complex(*posterior_y))]
    anterior_y = [*cmath.polar(complex(*anterior_y))]
    ventral_x = [*cmath.polar(complex(*ventral_x))]

    # 2.2 Rotation operation
    tail_head_phi = anterior_y[1]
    pos_y_phi = tail_head_phi - cmath.pi / 2
    neurons = [[[pt[0], pt[1] - pos_y_phi, pt[2]] for pt in pts] for pts in neurons]
    posterior_y[1] = posterior_y[1] - pos_y_phi
    anterior_y[1] = anterior_y[1] - pos_y_phi
    ventral_x[1] = ventral_x[1] - pos_y_phi

    # 2.3 Coordinate transformation: Polar -> Cartesian: (rho, phi, z) -> (x, y, z), (rho, phi) -> (x, y)
    neurons = [[[cmath.rect(pt[0], pt[1]).real, cmath.rect(pt[0], pt[1]).imag, pt[2]] for pt in pts] for pts in neurons]
    ventral_pt = [cmath.rect(*ventral_x).real, cmath.rect(*ventral_x).imag]
    # anterior_y = [cmath.rect(*anterior_y).real, cmath.rect(*anterior_y).imag]
    # posterior_y = [cmath.rect(*posterior_y).real, cmath.rect(*posterior_y).imag]

    # 3 Flip ventral-dorsal direction into x-axis positive direction
    neurons = [[[-pt[0], pt[1], pt[2]] for pt in pts] for pts in neurons] if ventral_pt[0] < 0 else neurons

    # 4 Robust transition
    transition_y = sum([sum([pt[1] for pt in pts]) / len(pts) for pts in neurons]) / len(neurons)
    neurons = [[[pt[0], pt[1] - transition_y, pt[2]] for pt in pts] for pts in neurons]
    # local count
    temp = [list(filter(lambda pt: abs(pt[1]) <= 10.0, pts)) for pts in neurons]
    transition_x = sum([sum([pt[0] for pt in pts]) / (len(pts) + 1e-5) for pts in temp]) / (len(temp) + 1e-5)
    neurons = [[[pt[0] - transition_x, pt[1], pt[2]] for pt in pts] for pts in neurons]

    return neurons
