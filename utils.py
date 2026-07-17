import numpy as np
import torch
from PIL import ImageFile
import heapq
from datasets.data_builder import build_dataloader

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import wandb


def train_dataset(train_function, config):
    if config["wandb"]:
        wandb.init(
            project=config["project"],
            config={c: a for c, a in config.items() if c != "data_config"},
            name=config["prefix"],
            mode="online",
            settings=wandb.Settings(start_method='thread')
        )
        wandb.define_metric("train_loss", step_metric="train_step")
        wandb.define_metric("test_loss", step_metric="test_step")
        wandb.define_metric("NF_samplewise_mean", step_metric="epoch")
        wandb.define_metric("NF_samplewise_max", step_metric="epoch")
        wandb.define_metric("NF_mean_image_roc", step_metric="epoch")
        wandb.define_metric("NF_pixel_roc", step_metric="epoch")
        wandb.define_metric("NF_aupro", step_metric="epoch")
        wandb.define_metric("NF_max_image_roc", step_metric="epoch")

    data_config = config["data_config"]
    train_loader, test_loader = build_dataloader(data_config, distributed=False)
    train_function(train_loader, test_loader, config=config)


class AnomalyTracker:

    def __init__(self, top_n=100):
        self.top_n = top_n
        self.anomalies = []
        self.normals = []

    def update(self, anomaly_score, filename, anomaly_map, gt_mask, label, image):
        if label == 1:
            entry = (-anomaly_score, filename, anomaly_map, gt_mask, image)
            if len(self.anomalies) < self.top_n:
                heapq.heappush(self.anomalies, entry)
            else:
                heapq.heappushpop(self.anomalies, entry)
        else:
            entry = (anomaly_score, filename, anomaly_map, gt_mask, image)
            if len(self.normals) < self.top_n:
                heapq.heappush(self.normals, entry)
            else:
                heapq.heappushpop(self.normals, entry)

    def get_top_anomalies(self):
        return sorted(self.anomalies, reverse=True)

    def get_top_normals(self):
        return sorted(self.normals, reverse=True)


def get_instancewise_data(data, config):
    labels, image, features = data["label"], data["image"], data["feature"]
    features = to_device([features], config["device"])[0]
    mask = data["mask"]
    img_in = features if config["pre_extracted"] else image
    cameras = data["camera"]

    return img_in, labels, image, mask, cameras, data["filename"]


def get_samplewise_data(data, config):
    B = data["feature_0"].shape[0]
    idx = torch.arange(B * 3)
    result = (idx % 3) * B + (idx // 3)

    labels = torch.cat(data["label"])[result]
    images = torch.cat([
        data["image_0"], data["image_1"], data["image_2"]
    ], dim=0)[result, ...]
    features = to_device([
        data["feature_0"], data["feature_1"], data["feature_2"]
    ], config["device"])
    masks = torch.cat([
        data["mask_0"], data["mask_1"], data["mask_2"]
    ], dim=0)[result, ...]

    B, C, H, W = features[0].shape
    foregrounds = torch.ones((3 * B, H, W)).to("cuda")

    filenames = np.concatenate(data["filename"])[result]
    cameras = torch.cat(data["cameras"])[result]

    return features, labels, images, masks, cameras, filenames, foregrounds



def t2np(tensor):
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def to_device(tensors, device):
    return [t.to(device) for t in tensors]


class Score_Observer:

    def __init__(self, name, percentage=True):
        self.name = name
        self.max_epoch = 0
        self.best_score = None
        self.last_score = None
        self.percentage = percentage

    def update(self, score, epoch, print_score=False):
        if self.percentage:
            score = score * 100
        self.last_score = score

        improved = False
        if epoch == 0 or score > self.best_score:
            self.best_score = score
            improved = True
        if print_score:
            self.print_score()
        return improved

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t best: {:.2f}'.format(
            self.name, self.last_score, self.best_score))


def model_size_info(model):
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(
        p.element_size() * p.numel()
        for p in model.parameters()
    ) / (1024 * 1024)

    output = f"**Model Size Info**\n"
    output += f"  * Number of Parameters: {num_params:,}\n"
    output += f"  * Model Size (MB): {model_size_mb:.2f} MB"
    return output


def save_weights(model, class_name, suffix, device="cuda"):
    save_to = "checkpoints"
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    model.to('cpu')
    torch.save(
        model.net.state_dict(),
        os.path.join(save_to, f'{class_name}_{suffix}.pth')
    )
    print('model saved')
    model.to(device)


def load_weights(model, class_name, suffix, device="cuda"):
    print("loading:", os.path.join("checkpoints", f'{class_name}_{suffix}.pth'))
    model.net.load_state_dict(
        torch.load(os.path.join("checkpoints", f'{class_name}_{suffix}.pth'))
    )
    model.eval()
    model.to(device)
    return model

