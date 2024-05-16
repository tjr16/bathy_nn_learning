# bathy_nn_learning

## Introduction

This repository is intended for loop closure detection and feature matching in the context of Multibeam Echo Sounders (MBES).

## Dependencies 

* Python 3.8
* PyTorch 1.10
* PyTorch Geometric 2.0
* opencv, wandb, shapely, open3d, [AUVLib (original repo)](https://github.com/nilsbore/auvlib)

Install the June 2022 version of AUVLib here:
    
    git clone -b extended_bm git@github.com:ignaciotb/auvlib.git

For details, please refer to `requirements.txt` (for pip) or `environment.yml` (for conda).

Recommended for baseline: PCL 1.10 or its [python binding](https://github.com/lijx10/PCLKeypoints)

## Data

Datasets are available for download here [datasets](https://drive.google.com/drive/folders/1UWxJw6cNCvzowqWpzo5eSEUT_0734tsG?usp=sharing)


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

## Citation

If you find our work useful, please consider citing:

    @inproceedings{tan2023data,
        title={Data-driven loop closure detection in bathymetric point clouds for underwater {SLAM}},
        author={Tan, Jiarui and Torroba, Ignacio and Xie, Yiping and Folkesson, John},
        booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
        pages={3131--3137},
        year={2023},
        organization={IEEE}
}
        
## Acknowledgment

Part of the code is based on some examples in [PyTorch Geometric](
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py).
