# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


URL: str = "https://osf.io/v2b5n/"
dataset = {
    "is_raw"              : True,  # False means synthetic data
    "state"               : "freely_moving",  # or straightened or immobile
    "color_info"          : False,
    "including_train_data": True,
    "including_test_data" : True,
    "num_animals"         : 3,
    "num_vols"            : 120,
    "animals"             : {
        "names"      : ["C1", "C2", "C3"],
        "name_reg"   : r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}",
        "ccords_root": "data/dataset/proofreading",

        "pt_clouds"  : {"root" : "data/dataset/label",
                        "train": [[["PR_v150_ImgStk001_dk001_{w6_Dt210514001057rebuild}_{red_from2966to3965}_Dt211119.mat", list(range(0, 40))]],
                                  [["PR_v051_ImgStk002_dk001_{w2_Dt210513220213Rebuild}_{red_from1516to2515}Dt210824.mat", list(range(1, 21))]],
                                  [["PR_v045_ImgStk001_dk001_{w4_Dt210513}_{Dt210824_new_PR_onDt211222}.mat", list(range(0, 20))]]],

                        "val"  : [[["PR_v150_ImgStk001_dk001_{w6_Dt210514001057rebuild}_{red_from2966to3965}_Dt211119.mat", list(range(40, 50))]],
                                  [["PR_v051_ImgStk002_dk001_{w2_Dt210513220213Rebuild}_{red_from1516to2515}Dt210824.mat", list(range(21, 26))]],
                                  [["PR_v045_ImgStk001_dk001_{w4_Dt210513}_{Dt210824_new_PR_onDt211222}.mat", list(range(20, 25))]]],

                        "test" : [[["PR_v006_ImgStk001_dk002_{w6_Dt210514001057rebuild}_{red_from4966to5965}}_Dt211123modified.mat", list(range(10))]],
                                  [["PR_v051_ImgStk002_dk001_{w2_Dt210513220213Rebuild}_{red_from1516to2515}Dt210824.mat", list(range(26, 31))]],
                                  [["PR_v045_ImgStk001_dk001_{w4_Dt210513}_{Dt210824_new_PR_onDt211222}.mat", list(range(25, 30))]]],

                        "long_term_test": []
                        },

        "fea_vecs"   : {"train_root": "data/benchmarks/CeNDeR1/train",
                        "val_root"  : "data/benchmarks/CeNDeR1/val",
                        "test_root" : "data/benchmarks/CeNDeR1/test"},

    }
}

engineering_feature = {
    "rec_z_scale"           : 1.0,  # xoy 1 pixel = 0.3 um, z axis 1 pixel = 5 um
    "rec_worm_diagonal_line": 400.0,
    "rec_knn_k"             : 40,
    "rec_des_len"           : 10,
    "unit"                  : 0.3,  # 1 pixel = 0.3 um
}

engineering_feature4fDNC = {
    "rec_z_scale"           : 1.0,
    "rec_worm_diagonal_line": 1.414,
    "rec_knn_k"             : 40,
    "rec_des_len"           : 10,
    "unit"                  : 0.3,  # 1 pixel = 0.3 um
    "ratio"                 : 0.3 / 84  # fdnc 1 pixel = 84 um
}
