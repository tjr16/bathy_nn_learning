"""Some parameters
"""

import math

## Dataset
CEREAL_PATH = "data/antarctica_2019.cereal"
XYZ_PATH = "data/antarctica_2019_relative.xyz"
# from which ping to which ping
TRAIN_PINGS = [36000, 74180]
TEST_PINGS = [74180, 77900]

## Dataset after preprocessing
TRAIN_BEAMS_PATH = "data/train_cloud.pcd"
TEST_BEAMS_PATH = "data/test_cloud.pcd"
TRAIN_TRAJ_PATH = "data/train_trajectory.bin"
TEST_TRAJ_PATH = "data/test_trajectory.bin"

## Dataset for training, validation and testing
TRAINING_PATH = "data/Circle100"
VALIDATION_PATH = "data/Circle100Valid"
TESTING_PATH = "data/Circle100Test"

## Dataset info
SIZE_SUBMAP = 8192  # num(points) in a submap
IOU_POSITIVE = [0.4, 0.8]  # IoU range for positive pairs

NUM_SUBMAPS = {
    "Circle100": 1500,     # training set
    "Circle100Valid": 400  # validation set
}

# Radius for circle datasets
RADIUS = {
    "submap": 100.0,
    "patch": 15.0
}


## Network config
CONFIG = {
    "SetAbstraction": {
        "max_num_neighbors": 64,
    },
    "PointNet2": {
        "r1": 10,
        "r2": 25,
        "dim_output": 32,
    },
    "Detector": {
        "ratio": 0.73,
    },
    "Descriptor": {
        "max_num_neighbors": 128,
        "dim_output": 32,
    },
    "Matcher": {
        "depth_period": math.pi/50,
    },
    "Margin": 1.0,  # margin in triplet loss
}

## Training config
PATH_CKPT = "save_models"
NUM_EPOCH = 50 # total training #epoch, default 100?
SAVE_EPOCH = 1  # save after each $SAVE_EPOCH$ epochs
BATCH_SIZE = 16 # 32 is old
NUM_WORKERS = 4
DATA_PATH = 'data/'

## Testing config
MODEL_PATH = None  # TODO
# parameters
KEYPOINT_THRESHOLD = 0.7  # threshold of keypoint detector
MATCHING_THRESHOLD = 0.1  # threshold of Eucledian distance
CORRECT_MATCH_DIST = 20   # (if there are ground truth labels)
DEPTH_FILTER_DIST = 2.0   # remove correspondences with depth difference larger than this
MATCH_NUM_THRESHOLD = 3   # positive: at least 3 feature correspondences
