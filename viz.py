import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
from pathlib import Path
import torch

def viz_roc(y_score=None, y_test=None, name=''):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.clf()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class ' + "NEEDS CLASSNAME")
    plt.legend(loc="lower right")
    plt.axis('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.tight_layout()
    plt.savefig(Path("viz", "roc", f"{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def compare_histogram(scores, classes, class_name, prefix, thresh=None, n_bins=64, log=False, name=''):
    if log:
        scores = np.log(scores + 1e-8)

    if thresh is not None:
        if np.max(scores) < thresh:
            thresh = np.max(scores)
        scores[scores > thresh] = thresh

    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 7))

    plt.rcParams.update({'font.size': 22})

    ax.hist(scores_norm, bins, alpha=0.5, density=True, label='Normal Samples', color='cyan', edgecolor="black", linewidth=1.2)
    ax.hist(scores_ano, bins, alpha=0.5, density=True, label='Defect Samples', color='crimson', edgecolor="black", linewidth=1.2)

    ticks = np.linspace(np.min(scores), np.max(scores), 5)
    labels = ['{:.2f}'.format(i) for i in ticks[:-1]] + ['>' + '{:.2f}'.format(np.max(scores))]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=16, weight='bold')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=16, weight='bold')

    ax.set_xlabel('Anomaly Score' if not log else 'Log Anomaly Score', fontsize=22, weight='bold')
    ax.set_ylabel('Density', fontsize=22, weight='bold')

    ax.legend(fontsize=22, frameon=True, edgecolor='black', prop={'weight': 'bold'})

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    dir_to_save = Path("viz", "hists", class_name, prefix)
    os.makedirs(dir_to_save, exist_ok=True)

    plt.savefig(Path(dir_to_save, f"{name}.png"), dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.close(fig)


def viz_maps(img, gt, ano_map, prefix='', norm=True, class_name=None, vmin=0, vmax=1,
             filename="test.png", title="sample_title"):

    ano_map = np.copy(ano_map)
    if True or norm:
        img = np.moveaxis(img, 0, 2)
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Input Image", fontsize=12, fontweight='bold')

    axs[1, 0].imshow(gt, vmin=0, vmax=1)
    axs[1, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')

    axs[1, 1].imshow(ano_map, vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("Anomaly Map", fontsize=12, fontweight='bold')

    axs[0, 1].axis('off')

    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[0, 0].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    dir_to_save = Path("viz", "maps", class_name, prefix)
    os.makedirs(dir_to_save, exist_ok=True)

    plt.savefig(Path(dir_to_save, filename), dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.close(fig)


def visualize(tracked_results, prefix, class_name, vmin, vmax, is_ano):

    for i, (score, filename, ano_map, gt, image) in enumerate(tracked_results):
        viz_maps(image, gt, ano_map, prefix, class_name=class_name, vmin=vmin, vmax=vmax,
                 filename=f"{'anomaly' if is_ano else 'normal'}_{i:04d}.png",
                 title=f"{filename}. {torch.round(score, decimals=4)}.")

