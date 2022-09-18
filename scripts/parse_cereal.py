"""
Step 1:
Parse .cereal dataset.
"""

import os

from auvlib.data_tools import std_data
from param import *
from utils.data import safe_save_pcd
from utils.dataset import collect_beams, collect_trajectory, random_sample

## Read cereal file
std_cereal = std_data.mbes_ping.read_data(CEREAL_PATH)

## Split training set and test set
train_map = std_cereal[TRAIN_PINGS[0]: TRAIN_PINGS[1]]
train_beams = collect_beams(train_map)
train_traj = collect_trajectory(train_map)

test_map = std_cereal[TEST_PINGS[0]: TEST_PINGS[1]]
test_beams = collect_beams(test_map)
test_traj = collect_trajectory(test_map)

## Write beams and trajectories into .pcd files
safe_save_pcd(train_beams, TRAIN_BEAMS_PATH, overwrite=False)
safe_save_pcd(test_beams, TEST_BEAMS_PATH, overwrite=False)
safe_save_pcd(train_traj, TRAIN_TRAJ_PATH, overwrite=False)
safe_save_pcd(test_traj, TEST_TRAJ_PATH, overwrite=False)

## If visualization is needed:
# train_beams_sample = random_sample(train_beams, 1e5)
# test_beams_sample = random_sample(test_beams, 1e5)

## Make directories for datasets
def make_dir(path):
    if not os.path.exists(path + "/raw/"):
        os.makedirs(path + "/raw/")
    if not os.path.exists(path + "/processed/"):
        os.makedirs(path + "/processed/")

make_dir(TRAINING_PATH)
make_dir(VALIDATION_PATH)
make_dir(TESTING_PATH)
