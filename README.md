# MG-Flow
=======

Code for **No angle overlooked: a multi-view normalizing flow framework for automated visual inspection of gear surface defects**.

MG-Flow is an unsupervised multi-view anomaly detection framework for automated visual inspection of gear surface defects. It trains only with defect-free samples and detects anomalous samples or regions through likelihood-based modeling of normal feature distributions with normalizing flows. The code supports experiments on a custom multi-view gear dataset and the public Real-IAD dataset.

The method is designed for practical gear inspection scenarios where complex three-dimensional geometry makes single-view observation incomplete, while parallax, local cross-view misalignment, edge shadows, and complex backgrounds can weaken defect responses. MG-Flow uses complementary structural information from multiple views to compensate for missing surface observations, and introduces a Foreground-focused Energy Selection Attention (FESA) module to enhance key gear structure responses while suppressing background noise.

For multi-view normalizing flow modeling, MG-Flow constructs a Multi-View Fusion Coupling (MVFC) block with a Cross-Block Scale-Translation (CBST) module. CBST performs cross-view block-level retrieval and intra-block fine-grained alignment inside the affine coupling layers, reducing misalignment noise and reliably integrating complementary local structures into normal feature distribution modeling. Experiments reported in the paper show strong detection, localization, and generalization performance on the custom multi-view gear dataset and Real-IAD.


## Installation

Create the environment:

```bash
conda env create --file environment.yml
conda activate mgflow
```

The code is mainly tested with CUDA. If your GPU or CUDA setting is different, please check the device setting in `config.py`.

## Data

Put the dataset under `data/`. The default paths are set in `config.py`:

```text
data/
data/realiad_jsons/
```

Each category should have a json file, for example:

```text
data/realiad_jsons/Cylindrical_Spur_Gear.json
```

If your data path is different, modify `config.py`.

## Preprocessing

Extract features before training:

```bash
python preprocess.py -c Cylindrical_Spur_Gear
```

The extracted files will be saved in `tmp/`.

## Training

Run the main model:

```bash
python train.py -c Cylindrical_Spur_Gear -arch cs_CBST -samplewise 1
```


## Results

The code reports image-level, sample-level and pixel-level anomaly detection results, including AUROC, AUPRO, AP and F1-max.

## Citation

If this code is useful for your research, please cite our paper after the final publication information is available.
>>>>>>> c2ebf97 (first commit)
