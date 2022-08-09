# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

URL = "https://osf.io/t7dzu/"
dataset_syn = {
    "is_raw"              : False,  # False means synthetic data
    "state"               : "freely_moving",  # or straightened or immobile
    "color_info"          : False,
    "including_train_data": True,
    "including_test_data" : True,
    "num_animals"         : 12,
    "num_vols"            : 236800,
    "animals"             : {
        "pt_clouds": {"train_root": "data/benchmarks/fDNC/train_synthetic",
                      "val_root"  : "data/benchmarks/fDNC/validation_synthetic",
                      "test_root" : "data/benchmarks/fDNC/test_synthetic"},
        "fea_vecs" : {"train_root": "data/benchmarks/fDNC/rec/feature_vectors1/train_synthetic",
                      "val_root"  : "data/benchmarks/fDNC/rec/feature_vectors1/validation_synthetic",
                      "test_root" : "data/benchmarks/fDNC/rec/feature_vectors1/test_synthetic"},
    },
}

benchmark_NeRVE = {
    "is_raw"              : True,  # False means synthetic data
    "state"               : "freely_moving",  # or straightened or immobile
    "color_info"          : False,
    "including_train_data": False,
    "including_test_data" : True,
    "num_animals"         : 1,
    "num_vols"            : 1372,  # is not consist with 1514 vols in NeRVE paper
    "animals"             : {"root"            : "data/benchmarks/fDNC/test_tracking",
                             "fea_vecs"        : {"4fDNC"         : "data/benchmarks/fDNC/fea_vec/test_tracking",
                                                  "4CeNDeR_cross" : "data/benchmarks/CeNDeR/test_tracking",
                                                  "4CeNDeR_within": "data/benchmarks/CeNDeR_within/test_tracking"},
                             "preprocessing"   : "data/benchmarks/fDNC/preprocessing/test_tracking",
                             "within_exp_setup": {"train": list(range(0, 40)),
                                                  "val"  : list(range(40, 50)),
                                                  "test" : list(range(50, 1372))},
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
                             "fea_vecs"     : {"4fDNC"         : "data/benchmarks/fDNC/fea_vec/test_neuropal_our",
                                               "4CeNDeR_cross" : "data/benchmarks/CeNDeR/test_neuropal_our",
                                               "4CeNDeR_within": "data/benchmarks/CeNDeR_within/test_neuropal_our"},
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
                             # "fea_vecs"     : "data/benchmarks/fDNC/rec/feature_vectors1/test_neuropal_Chaudhary",
                             "fea_vecs"     : {"4fDNC"         : "data/benchmarks/fDNC/fea_vec/test_neuropal_Chaudhary",
                                               "4CeNDeR_cross" : "data/benchmarks/CeNDeR/test_neuropal_Chaudhary",
                                               "4CeNDeR_within": "data/benchmarks/CeNDeR_within/test_neuropal_Chaudhary"},
                             "preprocessing": "data/benchmarks/fDNC/preprocessing/test_neuropal_Chaudhary"},
}

# Data and Network configs
engineering_feature = {
    "rec_worm_diagonal_line": 1.414,
    "rec_knn_k"             : 40,
    "rec_des_len"           : 10,
    "unit"                  : 84,  # 1 unit = 84 um
}

engineering_feature4CeNDeR = {
    "rec_worm_diagonal_line": 400.0,
    "rec_knn_k"             : 40,
    "rec_des_len"           : 10,
    "unit"                  : 84,  # 1 unit = 84 um
    "ratio"                 : 84 / 0.3  # CeNDeR 1 unit = 0.3 um
}
