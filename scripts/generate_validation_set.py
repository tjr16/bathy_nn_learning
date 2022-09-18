"""
Step 2-2:
Generate a validation set.
"""

import numpy as np
import open3d as o3d
from param import XYZ_PATH

pcd = o3d.io.read_point_cloud(XYZ_PATH)
pcd_np = np.asarray(pcd.points)
if pcd_np.shape[0] < 1e6:
    raise Exception(
        "Something wrong with point cloud reading! \
         Matplotlib can reduce the resolution of Open3D. So weird. \
         ")

import math

import torch
from param import *
from shapely.geometry import Polygon
from utils.data import read_bin, safe_save_pickle, save_bin
from utils.dataset import *

## Give a polygon for validation set
MIN_XY = (-math.inf, 4150)
MAX_XY = (math.inf, 6500)
pcd_np_select = select_region_square(pcd_np, min_xy=MIN_XY, max_xy=MAX_XY)

bounds_x = [12081, 13227, 15000, 13221, 12081]
bounds_y = [5686, 5148, 5600, 5903, 5686]
bounds_points = list(zip(bounds_x[:-1], bounds_y[:-1]))

## Sample submaps centers
num_sample = 3000
num_centers = NUM_SUBMAPS["Circle100Valid"]
fps_ratio = (num_centers-0.1)/ num_sample
poly = Polygon(bounds_points)
centers = sample_point_in_polygon(poly, n_points=num_sample)
centers_sample = fps_sample(torch.tensor(centers), sample_ratio=fps_ratio)

## Intersection over Union (IOU)
IOU_max, IOU_min = max(IOU_POSITIVE), min(IOU_POSITIVE)
num_centers = centers_sample.shape[0]
IOU_mat = np.zeros((num_centers, num_centers))
for i in range(0, num_centers):
    for j in range(i+1, num_centers):  # j > i always
        IOU_mat[i, j] = calculate_circle_IOU(r=RADIUS["submap"], 
            centerA=centers_sample[i, :], centerB=centers_sample[j, :])

## Create raw dataset: bin files
raw_path = VALIDATION_PATH + '/raw/'
processed_path = VALIDATION_PATH + '/processed/'
for idx, center in enumerate(centers_sample):
    crop_circle_and_save(pcd_np_select, center=center, path=raw_path, idx=idx)
print(f"{idx+1} submaps in total.")

# count the number of points in each submap
n_points_list = []
for idx in range(num_centers):
    temp = read_bin(raw_path + f"{idx}.bin")
    n_points_list.append(temp.shape[0]-1)  # excluding center point
print(f"num of points: min {min(n_points_list)}, max {max(n_points_list)}")

# collect positive submap pairs
positive_pairs = np.where((IOU_mat >= IOU_min) & (IOU_mat < IOU_max))
positive_pair_list = []
for first, second in zip(positive_pairs[0], positive_pairs[1]):
    positive_pair_list.append((first, second))
    positive_pair_list.append((second, first))

positive_pair_path = processed_path + 'positive_pair_list.pkl'
safe_save_pickle(positive_pair_list, positive_pair_path)

## Downsample and save as processed dataset
num_sample1 = 20000  # determined by min(n_points_list)
num_sample2 = SIZE_SUBMAP
fps_ratio =  (num_sample2-0.1)/num_sample1

for idx in range(NUM_SUBMAPS["Circle100Valid"]):
    read_path = raw_path + f"{idx}.bin"
    save_path = processed_path + f'processed_{idx}.bin'
    raw_bin = read_bin(read_path)
    raw_bin_center = raw_bin[0]
    raw_bin = raw_bin[1:] 
    processed_bin = random_sample(raw_bin, num_sample1)
    processed_bin = fps_sample(processed_bin, sample_ratio=fps_ratio)
    assert processed_bin.shape[0] == num_sample2
    processed_bin = np.concatenate([raw_bin_center.reshape(1, 3), processed_bin])
    save_bin(processed_bin, to_path=save_path)
