import os
from tqdm import tqdm
import copy
import argparse

import torch
from models.extractor import MultiFlowBackbone

import numpy as np
from utils import t2np, build_dataloader
from config import dataset


def extract_image_features(classname_to_do, extract_layer=35):
    model = MultiFlowBackbone()
    model.to("cuda")
    model.eval()

    config = {copy.copy(a): copy.copy(v) for (a, v) in dataset.items()}

    train_meta = config["train"]["meta_file"]
    test_meta = config["train"]["meta_file"]
    config["input_size"] = (768, 768)

    for class_name in [classname_to_do]:
        config["batch_size"] = 32
        config["train"]["meta_file"] = f"{train_meta}/{class_name}.json"
        config["test"]["meta_file"] = f"{test_meta}/{class_name}.json"
        config["classname"] = class_name
        config["type"] = "explicit"

        train_loader, test_loader = build_dataloader(config, distributed=True)

        model.to("cuda")
        for name, loader in zip(['train', 'test'], [train_loader, test_loader]):
            features = list()
            for i, data in enumerate(tqdm(loader)):
                img = data["image"].to("cuda")

                with torch.no_grad():
                    z = model(img)
                features.append(t2np(z))

            features = np.concatenate(features, axis=0)
            export_dir = os.path.join(config["feature_dir"], class_name)
            os.makedirs(export_dir, exist_ok=True)
            print(f"Saving features of shape {features.shape} to {export_dir}")
            np.save(os.path.join(export_dir, f'{name}.npy'), features)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "-classname", metavar="c", type=str,
                    default="Cylindrical_Spur_Gear",
                    help="Class name to process")
args, extras = parser.parse_known_args()

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    extract_image_features(args.c)

