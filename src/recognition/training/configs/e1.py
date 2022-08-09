# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

dataset = {
    "is_raw"              : True,  # False means synthetic data
    "state"               : "freely_moving",  # or straightened or immobile
    "color_info"          : False,
    "including_train_data": True,
    "including_test_data" : True,
    "num_animals"         : 3,
    "num_vols"            : 410,
    "animals"             : {
        "names"        : ["C1", "C2", "C3"],
        "name_reg"     : r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}",
        "ccords_root"  : "data/dataset/proofreading",
        "label_root"   : "data/dataset/label",
        "label"        : {
            "C1": [["PR_v150_ImgStk001_dk001_{w6_Dt210514001057rebuild}_{red_from2966to3965}_Dt211119.mat", list(range(50))],
                   ["PR_v012_ImgStk001_dk002_{w6_Dt210514001057rebuild}_{red_from4966to5965}.mat", list(range(50))],
                   ["PR_v011_ImgStk003_dk001_{w6_Dt210514001057rebuild}_{red_from6966to7965}.mat", list(range(50))],
                   ["PR_v008_ImgStk004_dk001_{w6_Dt210514001057rebuild}_{red_from7966to8965}.mat", list(range(50))],
                   ["PR_v008_ImgStk005_dk001_{w6_Dt210514001057rebuild}_{red_from8966to9965}.mat", list(range(50))],
                   ["PR_v010_ImgStk007_dk001_{w6_Dt210514001057rebuild}_{red_from10966to11965}.mat", list(range(50))],
                   ["PR_v013_ImgStk009_dk001_{w6_Dt210514001057rebuild}_{red_from12966to13965}.mat", list(range(50))]
                   ],
            "C2": [["PR_v051_ImgStk002_dk001_{w2_Dt210513220213Rebuild}_{red_from1516to2515}Dt210824.mat", list(range(1, 31, 1))]],
            "C3": [["PR_v045_ImgStk001_dk001_{w4_Dt210513}_{Dt210824_new_PR_onDt211222}.mat", list(range(30))]],
        },
        "fea_vecs_root": "data/benchmarks/supp/e1/",

    }
}

benchmark_NeRVE = {
    "is_raw"              : True,  # False means synthetic data
    "state"               : "freely_moving",  # or straightened or immobile
    "color_info"          : False,
    "including_train_data": False,
    "including_test_data" : True,
    "num_animals"         : 1,
    "num_vols"            : 1372,  # is not consist with 1514 vols in NeRVE paper
    "animals"             : {"root"         : "data/benchmarks/fDNC/test_tracking",
                             "fea_vecs"     : "data/benchmarks/supp/e1/test_tracking",
                             "preprocessing": "data/benchmarks/fDNC/preprocessing/test_tracking"},
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
                             "fea_vecs"     : "data/benchmarks/supp/e1/test_neuropal_our",
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
                             "fea_vecs"     : "data/benchmarks/supp/e1/test_neuropal_Chaudhary",
                             "preprocessing": "data/benchmarks/fDNC/preprocessing/test_neuropal_Chaudhary"},
}

method = {
    "name"                     : "CeNDeR",
    "network"                  : {},
    "fea_vecs_setup"           : {
        "rec_z_scale"           : 5,
        "rec_worm_diagonal_line": 400.0,
        "rec_knn_k"             : 25,
        "rec_des_len"           : 20,
        "unit"                  : 0.3,
    },
    "fea_ves_setup_fDNC4CeNDeR": {
        "rec_worm_diagonal_line": 400.0,
        "rec_knn_k"             : 25,
        "rec_des_len"           : 20,
        "unit"                  : 84,  # 1 unit = 84 um
        "ratio"                 : 84 / 0.3  # CeNDeR 1 unit = 0.3 um
    }
}
