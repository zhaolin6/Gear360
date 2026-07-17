import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import argparse
import wandb
from models.cv_model import Model
import config

from utils import (
    AnomalyTracker,
    Score_Observer,
    model_size_info,
    t2np,
    train_dataset,
    get_instancewise_data,
    get_samplewise_data,
    save_weights
)

from viz import (
    compare_histogram,
    visualize
)

from adeval import EvalAccumulatorCuda


def train(train_loader, test_loader, config):
    samplewise = config["data_config"]["samplewise"] == 1

    model = Model(config=config)
    model.to(config["device"])

    optimizer = torch.optim.AdamW(model.net.parameters(), lr=config["lr"], eps=1e-08,
                                  weight_decay=1e-5, betas=(0.9, 0.95))

    print(model_size_info(model))

    get_data = get_samplewise_data if samplewise else get_instancewise_data

    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')
    samplewise_mean_nll_obs = Score_Observer('Sample-wise AUROC mean over maps')
    samplewise_max_nll_obs = Score_Observer('Sample-wise AUROC  max over maps')
    pixel_st_obs = Score_Observer('AUROC pixel-wise')
    aupro_st_obs = Score_Observer('AUPRO for segmentation')

    mean_ap_obs = Score_Observer('AP mean over maps')
    max_ap_obs = Score_Observer('AP max over maps')
    samplewise_mean_ap_obs = Score_Observer('Sample-wise AP mean over maps')
    samplewise_max_ap_obs = Score_Observer('Sample-wise AP max over maps')

    mean_f1_max_obs = Score_Observer('F1-max mean over maps')
    max_f1_max_obs = Score_Observer('F1-max max over maps')
    samplewise_mean_f1_max_obs = Score_Observer('Sample-wise F1-max mean over maps')
    samplewise_max_f1_max_obs = Score_Observer('Sample-wise F1-max max over maps')

    train_iter, test_iter = 0, 0
    train_clamp = (torch.inf, -torch.inf)

    for epoch in range(config["meta_epochs"]):
        model.train()
        if config["verbose"]:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in tqdm(range(config["sub_epochs"])):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=config["hide_tqdm_bar"])):
                optimizer.zero_grad()
                img_in, labels, image, mask, cameras, filenames, foregrounds = get_data(data, config)
                z, jac = model(img_in)
                loss = model.loss(z, jac, mask=foregrounds)

                if config["wandb"]:
                    wandb.log({"train_loss": loss.item(), "train_step": train_iter})
                    train_iter += 1
                train_loss.append(t2np(loss))

                cat = torch.cat(z).detach().cpu()
                train_clamp = (min(train_clamp[0], torch.amin(cat).item()), max(train_clamp[1], torch.amax(cat).item()))

                loss.backward()
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if config["verbose"] and sub_epoch % 4 == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        accum = EvalAccumulatorCuda(train_clamp[0], train_clamp[1], train_clamp[0], train_clamp[1],
                                    nstrips=10000)
        print("TRAIN CLAMPS:", train_clamp)

        model.eval()
        if config["verbose"]:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_labels = list()
        img_nll = list()
        max_nlls = list()

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=config["hide_tqdm_bar"])):
                img_in, labels, image, mask, cameras, filenames, foregrounds = get_data(data, config)
                z, jac = model(img_in)

                loss = model.loss(z, jac, mask=foregrounds, per_sample=True)
                nll = model.loss(z, jac, mask=foregrounds, per_pixel=True)

                if nll.amin() < train_clamp[0]:
                    print(i, "Warning: Clamping outside of min", nll.amin())
                if nll.amax() > train_clamp[1]:
                    print(i, "Warning: Clamping outside of max", nll.amax())

                ano_map = torch.nn.functional.interpolate(nll.unsqueeze(1), (964, 964), mode="bilinear")
                ano_map = torch.clamp(ano_map.squeeze(), train_clamp[0], train_clamp[1]).cuda(non_blocking=True)
                mask = mask.to(torch.uint8).squeeze().cuda(non_blocking=True)

                img_score = torch.amax(nll, dim=(-1, -2))

                accum.add_anomap_batch(ano_map, mask)
                accum.add_image(torch.clamp(img_score, train_clamp[0], train_clamp[1]), labels)

                img_nll.append(t2np(loss))
                max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))
                test_loss.append(loss.mean().item())
                if config["wandb"]:
                    wandb.log({"test_loss": test_loss[-1], "test_step": test_iter})
                    test_iter += 1
                test_labels.append(labels)

        img_nll = np.concatenate(img_nll)
        max_nlls = np.concatenate(max_nlls)
        test_loss = np.mean(np.array(test_loss))

        if config["verbose"]:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        sample_wise_labels = test_labels.reshape((-1, 3)).any(axis=1)
        instance_wise_scores_max = max_nlls.reshape((-1, 3)).mean(axis=1)
        instance_wise_scores_mean = img_nll.reshape((-1, 3)).mean(axis=1)

        for lbl, score in zip(sample_wise_labels, instance_wise_scores_max):
            accum.add_sample(score, lbl)

        print(accum.summary())

        metrics = accum.summary()
        if epoch == config["meta_epochs"] - 1:
            compare_histogram(img_nll, test_labels, config["class_name"], config["prefix"], name=f"imagewise_mean",
                              thresh=3)
            compare_histogram(max_nlls, test_labels, config["class_name"], config["prefix"], name="imagewise_max",
                              thresh=3)
            compare_histogram(instance_wise_scores_mean, sample_wise_labels, config["class_name"],
                              config["prefix"], name="samplewise_mean", thresh=3, n_bins=64)
            compare_histogram(instance_wise_scores_max, sample_wise_labels, config["class_name"],
                              config["prefix"], name="samplewise_max", thresh=3, n_bins=64)

        def calculate_f1_max(scores, labels):
            thresholds = np.linspace(scores.min(), scores.max(), 1000)
            f1_scores = []

            for threshold in thresholds:
                predictions = (scores >= threshold).astype(int)
                f1 = f1_score(labels, predictions, zero_division=0)
                f1_scores.append(f1)

            return np.max(f1_scores) if f1_scores else 0.0

        mean_nll_obs.update(roc_auc_score(is_anomaly, img_nll), epoch,
                            print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch,
                           print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_mean_nll_obs.update(roc_auc_score(sample_wise_labels, instance_wise_scores_mean), epoch,
                                       print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_max_nll_obs.update(roc_auc_score(sample_wise_labels, instance_wise_scores_max), epoch,
                                      print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)

        aupro_st_obs.update(metrics["p_aupro"], epoch, True)
        pixel_st_obs.update(metrics["p_auroc"], epoch, True)

        mean_ap_obs.update(average_precision_score(is_anomaly, img_nll), epoch,
                           print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        max_ap_obs.update(average_precision_score(is_anomaly, max_nlls), epoch,
                          print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_mean_ap_obs.update(average_precision_score(sample_wise_labels, instance_wise_scores_mean), epoch,
                                      print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_max_ap_obs.update(average_precision_score(sample_wise_labels, instance_wise_scores_max), epoch,
                                     print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)

        mean_f1_max_obs.update(calculate_f1_max(img_nll, is_anomaly), epoch,
                               print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        max_f1_max_obs.update(calculate_f1_max(max_nlls, is_anomaly), epoch,
                              print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_mean_f1_max_obs.update(calculate_f1_max(instance_wise_scores_mean, sample_wise_labels), epoch,
                                          print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_max_f1_max_obs.update(calculate_f1_max(instance_wise_scores_max, sample_wise_labels), epoch,
                                         print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)

        if config["wandb"]:
            wandb.log({
                "epoch": epoch,
                "NF_samplewise_mean": samplewise_mean_nll_obs.last_score,
                "NF_samplewise_max": samplewise_max_nll_obs.last_score,
                "NF_mean_image_roc": mean_nll_obs.last_score,
                "NF_pixel_roc": pixel_st_obs.last_score,
                "NF_aupro": aupro_st_obs.last_score,
                "NF_max_image_roc": max_nll_obs.last_score,
                "NF_mean_AP": mean_ap_obs.last_score,
                "NF_max_AP": max_ap_obs.last_score,
                "NF_samplewise_mean_AP": samplewise_mean_ap_obs.last_score,
                "NF_samplewise_max_AP": samplewise_max_ap_obs.last_score,
                "NF_mean_F1_max": mean_f1_max_obs.last_score,
                "NF_max_F1_max": max_f1_max_obs.last_score,
                "NF_samplewise_mean_F1_max": samplewise_mean_f1_max_obs.last_score,
                "NF_samplewise_max_F1_max": samplewise_max_f1_max_obs.last_score
            })
        accum.reset()

    if config["save_model"]:
        save_weights(model, config["class_name"], config["prefix"], config["device"])

    return mean_nll_obs, max_nll_obs, pixel_st_obs, aupro_st_obs, \
           mean_ap_obs, max_ap_obs, samplewise_mean_ap_obs, samplewise_max_ap_obs, \
           mean_f1_max_obs, max_f1_max_obs, samplewise_mean_f1_max_obs, samplewise_max_f1_max_obs


if __name__ == "__main__":
    arch_choices = [
        "cs_naive",
        "cs_neigh",
        "cs_att_cross",
        "cs_att_self",
        "cs_CBST",
        "cs_STVC",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", metavar="p", type=str,
                        default="to_delete")
    parser.add_argument("-project", type=str, default="03_csflow_realiad")
    parser.add_argument("-seed", type=int, default=10000)
    parser.add_argument("-wandb", type=int, default=0)
    parser.add_argument("-save_model", type=int, default=0)
    parser.add_argument("-use_noise", type=int, default=1)
    parser.add_argument("-c", "-classname", metavar="c", type=str,
                        default="Cylindrical_Spur_Gear")
    parser.add_argument("-multi", metavar="b", type=int,
                        help="0 for regular training. 1 for multi-class",
                        default=0)
    parser.add_argument("-arch", type=str, help="Chose type of architecture", choices=arch_choices,
                        default="cs_CBST")
    parser.add_argument("-samplewise", metavar="b", type=int,
                        help="0 for mixing the training images freely. 1 for sorting them by instance/sample",
                        default=1)
    parser.add_argument("-show_heatmap", type=int, default=0,
                        help="Whether to show/save heatmap during testing")

    args, extras = parser.parse_known_args()

    print(args)
    assert args.samplewise if "cs" in args.arch else not args.samplewise, f"For CS-Flow architecture {args.arch}, samplewise dataloading is needed!"

    class_name = args.c
    config_obj = config.effnet_config

    config_obj["prefix"] = args.prefix
    config_obj["project"] = args.project
    config_obj["wandb"] = args.wandb == 1
    config_obj["arch"] = args.arch
    config_obj["class_name"] = class_name
    config_obj["multi"] = args.multi
    config_obj["seed"] = args.seed
    config_obj["use_noise"] = args.use_noise
    config_obj["data_config"]["samplewise"] = args.samplewise
    config_obj["save_model"] = args.save_model == 1
    config_obj["show_heatmap"] = True
    config_obj["show_heatmap"] = args.show_heatmap == 1

    torch.manual_seed(args.seed)

    config_obj["data_config"]["train"][
        "meta_file"] = f"{config_obj['data_config']['train']['meta_file']}/{class_name}.json"
    config_obj["data_config"]["test"][
        "meta_file"] = f"{config_obj['data_config']['test']['meta_file']}/{class_name}.json"
    config_obj["data_config"]["classname"] = class_name

    if args.samplewise:
        config_obj["data_config"]["batch_size"] = 8
    else:
        raise NotImplementedError("This code & data-loading has to be executed sample-wise to work properly.")

    print(f"Executing for classname: {class_name}")
    train_dataset(train, config=config_obj)

