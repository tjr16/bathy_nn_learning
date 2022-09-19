# bathy_nn_learning

## Introduction

This repository is intended for loop closure detection and feature matching in the context of Multibeam Echo Sounders (MBES).

## Dependencies 

* Python 3.8
* PyTorch 1.10
* PyTorch Geometric 2.0
* opencv, wandb, shapely, open3d, [AUVLib](https://github.com/nilsbore/auvlib)

For details, please refer to `requirements.txt` (for pip) or `environment.yml` (for conda).

Recommended for baseline: PCL 1.10 or its [python binding](https://github.com/lijx10/PCLKeypoints)

## Usage

Step 1: Run `scripts/parse_cereal.py` to parse cereal data.

Step 2.1 - 2.3: Run other scripts in `scripts/` to create datasets.

Step 3: Run `train.py` to train a model. (Modify `param.py` properly.)

Step 4: Run scripts in `test/` to evaluate the model.

    .
    ├── data               # datasets
    │   ├── Circle100      # training set
    │   │   ├── raw        # raw training set
    │   │   └── processed  # processed training set
    │   ├── Circle100Valid
    │   │   └── ...
    │   └── Circle100Test
    │       └── ...
    ├── scripts     # scripts for data processing
    ├── utils       # utility functions 
    ├── test        # testing scripts
    ├── models.py   # model implementation
    ├── dataset.py  # dataset implementation
    ├── param.py    # parameters and configurations
    └── train.py    # training script

## Acknowledgment

Part of the code is based on some examples in [PyTorch Geometric](
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py).
