# REALIAD-SETTINGS
dataset = {
    "feature_dir": 'tmp/',
    "type": "explicit",
    "image_reader": {
        "type": "opencv",
        "kwargs": {
            "image_dir": "data/",
            "color_mode": "RGB",
        }
    },
    "input_size": (964, 964),
    "pixel_mean": (0.485, 0.456, 0.406),
    "pixel_std": (0.229, 0.224, 0.225),
    "test": {
        "meta_file": "data/realiad_jsons"
    },
    "train": {
        "hflip": False,
        "rebalance": False,
        "rotate": False,
        "vflip": False,
        "meta_file": "data/realiad_jsons"
    },
    "type": "feature",
    "workers": 0,
    "batch_size": 32,
}

effnet_config = {
    "n_coupling_blocks": 6,
    "img_len": 768,
    "data_config": dataset,
    "type": None
}

effnet_config.update({
    "pre_extracted": True,
    "device": "cuda",

    "img_size": (effnet_config["img_len"], effnet_config["img_len"]),
    "img_dims": [3, effnet_config["img_len"], effnet_config["img_len"]],
    "map_len": effnet_config["img_len"] // 32,
    "extract_layer": 35,
    "img_feat_dims": 304,
    "n_feat": 304,
    "pos_enc": 0,

    "depth_len": None,
    "depth_channels": None,
    "depth_channels": None,

    "clamp": 1.9,
    "channels_hidden_teacher": 64,
    "channels_hidden_student": None,
    "use_gamma": True,
    "kernel_sizes": [3] * (effnet_config["n_coupling_blocks"] - 1) + [5],

    "verbose": True,
    "hide_tqdm_bar": True,
    "save_model": False,

    "lr": 2e-4,
    "batch_size": 8,
    "eval_batch_size": 16,
    "meta_epochs": 15,
    "sub_epochs": 4,
    "use_noise": 0,
})

