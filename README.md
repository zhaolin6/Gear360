
# Gear360

This repository provides the official implementation of **Gear360: A Multi-View Normalizing Flow Framework for Full-Surface Gear Defect Detection**. Gear360 is designed for full-surface gear defect detection under multi-view visual inspection, where a single view may fail to cover all critical surface regions and multiple views may suffer from parallax, local misalignment, background interference, and illumination variations. Trained only on defect-free samples, Gear360 exploits complementary structural information from multi-view images and models normal feature distributions with normalizing flows for sample-level defect discrimination and pixel-level defect localization. Specifically, a Foreground-focused Energy Selection Attention (FESA) module is introduced to enhance gear structure-related responses and suppress background interference. A Multi-View Fusion Coupling (MVFC) block with a Cross-Block Scale-Translation (CBST) module is further designed to perform cross-view block-level retrieval and intra-block fine-grained alignment, thereby reducing cross-view misalignment and incorporating complementary multi-view structural information into affine coupling parameter prediction. The framework is evaluated on our custom three-view gear surface defect dataset and the public five-view Real-IAD dataset, demonstrating strong detection and localization performance across different industrial objects and view configurations. Gear360 contains only **13.49 M parameters** and achieves an inference speed of **52.07 FPS**, showing good potential for online industrial visual inspection.

## Installation

Create the environment:

```bash
conda env create --file environment.yml
conda activate Gear360
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
