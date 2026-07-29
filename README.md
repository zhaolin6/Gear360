# MG-Flow
This repository provides the official implementation of **MG-Flow**, an unsupervised multi-view anomaly detection framework for industrial surface defect inspection.
MG-Flow is designed to learn normal feature distributions from defect-free samples and identify anomalous samples or regions through likelihood-based modeling with normalizing flows. By using multi-view images, the method can exploit complementary structural information from different viewpoints, which helps improve inspection stability when single-view observations are incomplete or affected by occlusion, viewpoint variation, local misalignment, shadows, or background interference.
The framework enhances foreground object features with the Foreground-focused Energy Selection Attention (FESA) module and performs multi-view feature distribution modeling through a normalizing flow architecture. To improve cross-view interaction, MG-Flow introduces a Multi-View Fusion Coupling (MVFC) block with a Cross-Block Scale-Translation (CBST) module, enabling block-level cross-view retrieval and fine-grained local alignment inside the affine coupling layers.
This code supports experiments on the custom multi-view industrial defect dataset used in our work and can also be adapted to public multi-view industrial anomaly detection datasets such as Real-IAD.

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
