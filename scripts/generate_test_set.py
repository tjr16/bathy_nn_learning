"""
Step 2-3:
Generate a test set.
"""

import warnings

import numpy as np
from auvlib.data_tools import std_data
from param import *
from scipy.spatial import distance_matrix
from utils.data import (numpy_to_pcd, pcd_to_numpy, read_bin, read_pcd,
                        safe_save_pickle, save_pcd)
from utils.dataset import (collect_beams, fps_sample, random_sample,
                           select_region_circle)

## Read cereal
std_cereal = std_data.mbes_ping.read_data(CEREAL_PATH)
train_map = std_cereal[TRAIN_PINGS[0]: TRAIN_PINGS[1]]
test_map = std_cereal[TEST_PINGS[0]: TEST_PINGS[1]]
raw_path = TESTING_PATH + '/raw/'
processed_path = TESTING_PATH + '/processed/'

## Generate positive pairs
positive_pair_path = raw_path + 'pairs.pkl'
new_positive_pair_path = processed_path + 'new_pairs.pkl'
# train (source) refers to the first pass trajectory 
# (but not included in training set)
# test (target) refers to the revisit trajectory
train_traj = read_bin(TRAIN_TRAJ_PATH)
test_traj = read_bin(TEST_TRAJ_PATH)

# sample from trajectories, one for every 10 pings
len_train = train_traj.shape[0]
len_test = test_traj.shape[0]
idx_train = np.arange(0, len_train, 10)
idx_test = np.arange(0, len_test, 10)
train_traj_sample = train_traj[idx_train, :]
test_traj_sample = test_traj[idx_test, :]

dist_mat = distance_matrix(train_traj_sample, test_traj_sample)
dist_mask = (dist_mat < 70) & (dist_mat > 20)

pairs = []
# for each sampled target submap in test set
for i in range(0, len(idx_test)):
    print(f"...{i}...")
    candidates = dist_mat[:, i]
    mask = dist_mask[:, i]
    mask_indices = np.nonzero(mask)
    len_mask = mask.sum()
    if len_mask == 0:
        continue
    # randomly sample one source submap from candidates in training set
    random_idx = np.random.randint(len_mask)
    return_idx = mask_indices[0][random_idx]
    pairs.append((int(10 * return_idx), 10 * i))

# save positive pairs (not the pairs finally used)
safe_save_pickle(pairs, positive_pair_path)

## Generate raw dataset
# indices of train/ test submaps
train_set = set(list(map(lambda x: x[0], pairs)))
test_set = set(list(map(lambda x: x[1], pairs)))

# train submaps
count_train_points, count_test_points = [], []
len_train_map, len_test_map = len(train_map), len(test_map)
for train_id in train_set:
    center_ping = train_map[train_id]
    center = center_ping.pos_
    center_xy = center[:-1]
    begin_index = int(train_id - 2 * RADIUS["submap"])
    end_index = int(train_id + 2 * RADIUS["submap"])
    if begin_index < 0: begin_index = 0
    if end_index > (len_train_map - 1): end_index = (len_train_map - 1)
    all_pings = train_map[begin_index: end_index]
    train_cloud = collect_beams(all_pings)
    cloud_np = select_region_circle(train_cloud, center=center_xy, radius=RADIUS["submap"])
    count_train_points.append(cloud_np.shape[0])
    cloud_np_with_center = np.concatenate((center.reshape(1, 3), cloud_np), axis=0)
    cloud_pcd = numpy_to_pcd(cloud_np_with_center)
    save_pcd(cloud_pcd, raw_path + f"train_{train_id}.pcd")
print(f"Count train points: min={min(count_train_points)}, max={max(count_train_points)}")
# min=23760, max=60173

# test submaps
for test_id in test_set:
    center_ping = test_map[test_id]
    center = center_ping.pos_
    center_xy = center[:-1]
    begin_index = int(test_id - 2 * RADIUS["submap"])
    end_index = int(test_id + 2 * RADIUS["submap"])
    if begin_index < 0: begin_index = 0
    if end_index > (len_test_map - 1): end_index = (len_test_map - 1)
    all_pings = test_map[begin_index: end_index]
    test_cloud = collect_beams(all_pings)
    cloud_np = select_region_circle(test_cloud, center=center_xy, radius=RADIUS["submap"])
    count_test_points.append(cloud_np.shape[0])
    cloud_np_with_center = np.concatenate((center.reshape(1, 3), cloud_np), axis=0)
    cloud_pcd = numpy_to_pcd(cloud_np_with_center)
    save_pcd(cloud_pcd, raw_path + f"test_{test_id}.pcd")
print(f"Count test points: min={min(count_test_points)}, max={max(count_test_points)}") # min=15151, max=58657

## Generate processed dataset
NUM_MAX = 50000  # abandon if exceed
NUM_MIN = 10000  # warn 
NUM_RANDOM = 30000
fps_ratio = (SIZE_SUBMAP-0.1)/NUM_RANDOM

count_train_points, count_test_points = [], []
bad_train_id, bad_test_id = [], []
for train_id in train_set:
    train_pcd = pcd_to_numpy(read_pcd(raw_path + f"train_{train_id}.pcd"))
    train_np = train_pcd[1:, :]  # without auv pose
    center = train_pcd[0, :]
    n_points = train_np.shape[0]  # without 
    if n_points < NUM_MIN:
        warnings.warn(f"Too few points in train_id={train_id}. \
                    If this happens, the raw folder might have been overwritten.")
    if (n_points > NUM_MAX) or (n_points < NUM_RANDOM):
        bad_train_id.append(train_id)
        continue
    if n_points >= NUM_RANDOM:
        train_np = random_sample(train_np, NUM_RANDOM)
    train_np = fps_sample(train_np, fps_ratio)
    count_train_points.append(n_points)
    train_np_processed = np.concatenate((center.reshape(1, 3), train_np), axis=0)
    train_pcd = numpy_to_pcd(train_np_processed)
    save_pcd(train_pcd, processed_path + f"train_{train_id}.pcd")

print(f"Count train points: min={min(count_train_points)}, max={max(count_train_points)}")

# test
for test_id in test_set:
    test_pcd = pcd_to_numpy(read_pcd(raw_path + f"test_{test_id}.pcd"))
    test_np = test_pcd[1:, :]  # without auv pose
    center = test_pcd[0, :]
    n_points = test_np.shape[0]  # without 
    if n_points < NUM_MIN:
        warnings.warn(f"Too few points in test_id={test_id}. \
                    If this happens, the raw folder might have been overwritten.")
    if (n_points > NUM_MAX) or (n_points < NUM_RANDOM):
        bad_test_id.append(test_id)
        continue
    if n_points >= NUM_RANDOM:
        test_np = random_sample(test_np, NUM_RANDOM)
    test_np = fps_sample(test_np, fps_ratio)
    count_test_points.append(n_points)
    test_np_processed = np.concatenate((center.reshape(1, 3), test_np), axis=0)
    test_pcd = numpy_to_pcd(test_np_processed)
    save_pcd(test_pcd, processed_path + f"test_{test_id}.pcd")

print(f"Count test points: min={min(count_test_points)}, max={max(count_test_points)}")

## Remove bad train/ test indices from pairs and create a new one
bad_pairs_id = []
for idx, (train_id, test_id) in enumerate(pairs):
    if (train_id in bad_train_id) or (test_id in bad_test_id):
        bad_pairs_id.append(idx)

new_pairs = np.array(pairs)
new_pairs = np.delete(new_pairs, bad_pairs_id, axis=0)
safe_save_pickle(new_pairs, new_positive_pair_path)
