"""
Step 2-1:
Generate a training set.
"""

import numpy as np
import open3d as o3d
from param import *
from shapely.geometry import Polygon
from utils.data import read_bin, safe_save_pickle, save_bin
from utils.dataset import *

pcd = o3d.io.read_point_cloud(TRAIN_BEAMS_PATH)
pcd_np = np.asarray(pcd.points)
if pcd_np.shape[0] < 1e6:
    raise Exception(
        "Some error happens with point cloud reading! \
         Matplotlib bothers Open3D. FIXME. \
         ")


## Give polygons for training set
bounds_x1 = [571933, 573337, 574691, 574505, 571933]
bounds_y1 = [1842514, 1846319, 1846883, 1846193, 1842514]
bounds_x2 = [574373, 574395, 573238, 572366, 574373]
bounds_y2 = [1844969, 1844932, 1841735, 1842144, 1844969]
bounds_points1 = list(zip(bounds_x1[:-1], bounds_y1[:-1]))
bounds_points2 = list(zip(bounds_x2[:-1], bounds_y2[:-1]))
bounds1 = (bounds_x1, bounds_y1)
bounds2 = (bounds_x2, bounds_y2)

## Sample submaps centers
random_num_centers = 4000
num_centers = NUM_SUBMAPS["Circle100"]
polys = [Polygon(bounds_points1), Polygon(bounds_points2)]
centers = sample_point_in_multiple_polygon(polys, n_points=random_num_centers)
centers_sample = fps_sample(np.array(centers), sample_ratio=num_centers/random_num_centers)  # return numpy
pcd_np_sample = random_sample(pcd_np, pcd_np.shape[0]//100)

## Intersection over Union (IOU)
IOU_max, IOU_min = max(IOU_POSITIVE), min(IOU_POSITIVE)
num_centers = centers_sample.shape[0]
IOU_mat = np.zeros((num_centers, num_centers))
# example: calculate_circle_IOU(r=100, centerA=np.array([0, 0]), centerB=np.array([0, 60]))
for i in range(0, num_centers):
    for j in range(i+1, num_centers):  # j > i always
        IOU_mat[i, j] = calculate_circle_IOU(r=RADIUS["submap"], centerA=centers_sample[i, :], centerB=centers_sample[j, :])


## Create raw dataset: .bin files
raw_path = TRAINING_PATH + '/raw/'
processed_path = TRAINING_PATH + '/processed/'
for idx, center in enumerate(centers_sample):
    crop_circle_and_save(pcd_np, center=center, path=raw_path, idx=idx)
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
num_sample1 = 19000  # determined by min(n_points_list)
num_sample2 = SIZE_SUBMAP
fps_ratio =  (num_sample2-0.1)/num_sample1

for idx in range(NUM_SUBMAPS["Circle100"]):
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
    
