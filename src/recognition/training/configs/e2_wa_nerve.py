# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

dataset = {
    "name"                : "NeRVE",
    "is_raw"              : True,  # False means synthetic data
    "state"               : "freely_moving",  # or straightened or immobile
    "color_info"          : False,
    "including_train_data": False,
    "including_test_data" : True,
    "num_animals"         : 1,
    "num_vols"            : 1372,  # is not consist with 1514 vols in NeRVE paper
    "animals"             : {"root": {"raw"          : "data/benchmarks/fDNC/test_tracking",
                                      "preprocessing": "data/benchmarks/fDNC/preprocessing/test_tracking",
                                      "fea_vecs"     : "data/benchmarks/supp/e2_wa_nerve"},
                             "set" : {"train": (0, 40),
                                      "val"  : (40, 50),
                                      "test" : (50, 1372)},
                             },
}

benchmark_NeuroPAL = {
    "is_raw"              : True,  # False means synthetic data
    "state"               : "immobile",  # or straightened
    "color_info"          : True,
    "including_train_data": False,
    "including_test_data" : True,
    "num_animals"         : 11,
    "num_vols"            : 11,
    "animals"             : {"root"         : "data/benchmarks/fDNC/test_neuropal_our",
                             "fea_vecs"     : "data/benchmarks/supp/test_neuropal_our",
                             "preprocessing": "data/benchmarks/fDNC/preprocessing/test_neuropal_our"},
}

benchmark_NeuroPAL_Chaudhary = {
    "is_raw"              : True,  # False means synthetic data
    "state"               : "immobile",  # or straightened or immobile
    "color_info"          : True,
    "including_train_data": False,
    "including_test_data" : True,
    "num_animals"         : 9,
    "num_vols"            : 9,
    "animals"             : {"root"         : "data/benchmarks/fDNC/test_neuropal_Chaudhary",
                             "fea_vecs"     : "data/benchmarks/supp/test_neuropal_Chaudhary",
                             "preprocessing": "data/benchmarks/fDNC/preprocessing/test_neuropal_Chaudhary"},
}

method = {
    "name"                     : "CeNDeR",
    "network"                  : {},
    "fea_vecs_setup"           : {
        "rec_z_scale"           : 5,
        "rec_worm_diagonal_line": 400.0,
        "rec_knn_k"             : 25,
        "rec_des_len"           : 10,
        "unit"                  : 0.3,

    },
    "fea_ves_setup_fDNC4CeNDeR": {
        "rec_z_scale"           : 1,
        "rec_worm_diagonal_line": 400.0,
        "rec_knn_k"             : 25,
        "rec_des_len"           : 10,
        "unit"                  : 84,  # 1 unit = 84 um
        "ratio"                 : 84 / 0.3  # CeNDeR 1 unit = 0.3 um
    }

}
id_map = {0 : 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 17: 15, 19: 16, 20: 17, 21: 18, 22: 19, 23: 20, 24: 21, 25: 22, 26: 23, 27: 24, 28: 25,
          29: 26, 30: 27, 31: 28, 32: 29, 33: 30, 35: 31, 36: 32, 38: 33, 39: 34, 40: 35, 41: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48,
          55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 61: 54, 62: 55, 63: 56, 64: 57, 65: 58, 66: 59, 67: 60, 68: 61, 70: 62, 71: 63, 72: 64, 73: 65, 74: 66, 75: 67, 76: 68, 77: 69, 78: 70, 79: 71,
          80: 72, 81: 73, -1: 74}

pi = list(id_map.keys())[:-1]
