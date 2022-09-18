
import itertools
import random
from math import inf

import numpy as np
import numpy.linalg as nl
import torch
from param import RADIUS
from shapely.geometry import Point
from torch_geometric.nn import fps

from .data import safe_load_pickle, save_bin, to_np


## === select points ===
def select_region_square(pcd_np, min_xy=None, max_xy=None):
    """Select a square region.
    """
    # pcd_np as numpy.ndarray has been copied!
    if min_xy:
        pcd_np = pcd_np[(pcd_np[:, 0] > min_xy[0]) & (pcd_np[:, 1] > min_xy[1])]
    if max_xy:
        pcd_np = pcd_np[(pcd_np[:, 0] < max_xy[0]) & (pcd_np[:, 1] < max_xy[1])]
    return pcd_np

def select_region_circle(pcd_np, center, radius):
    # NOTE: center is unnecessary to be the mean value of all points!
    pcd_np_xy = pcd_np[:, :-1]
    # pcd_np as numpy.ndarray has been copied!
    pcd_np = pcd_np[np.linalg.norm(pcd_np_xy-center, axis=1) < radius]
    return pcd_np

def crop_circle_and_save(pcd: np.ndarray, center: np.ndarray, path:str, idx: int):
    """Create circle submaps for training and validation sets.
    The additional point as the first one in a submap is its center.

    Args:
        pcd (np.ndarray): The entire point cloud.
        center (np.ndarray): A submap center where the submap should be sampled.
        path (str): File save path.
        idx (int): The index of the submap.
    """
    pcd_crop = select_region_circle(
        pcd, center=center, radius=RADIUS["submap"]
        )  # crop a circle
    center_xyz = np.zeros((1, 3))
    center_xyz[:, :2] = center.reshape((1, 2))  # center.x and center.y
    center_xyz[:, 2] = pcd_crop[:, -1].mean()   # mean depth as center.z 
    if idx % 10 == 0:
        print(f"Submap {idx} shape: {pcd_crop.shape}")
    pcd_crop_with_center = np.concatenate([center_xyz, pcd_crop])  # (1+N, 3)
    save_bin(pcd_np=pcd_crop_with_center, to_path=path + f"{idx}.bin")


## === sampling ===
def random_sample(pcd_np: np.ndarray, n_points):
    pcd_np_sample = pcd_np[np.random.choice(pcd_np.shape[0], size=int(n_points), replace=False), :]
    return pcd_np_sample

def fps_sample(pcd_np: np.ndarray, sample_ratio):
    """Use Farthest Point Sampling (FPS) to sample。
    """
    if pcd_np.shape[0] > 1e5:
        raise Exception("Too large point cloud! Errors may arise, so randomly downsample it first.")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if type(pcd_np) is not torch.Tensor:
        pcd_tensor = torch.tensor(pcd_np).to(device)
    else:
        pcd_tensor = pcd_np.to(device)
    idx = fps(pcd_tensor, ratio=sample_ratio)  # rounded up
    pcd_np_sample = pcd_tensor[idx, :].cpu().numpy()
    return pcd_np_sample

def sample_point_in_polygon(poly, n_points=None):
    x_min, y_min, x_max, y_max = poly.bounds
    if n_points is None:
        while True:
            p = Point(random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            if poly.contains(p):
                    return (p.x, p.y)
    else:
        points = []
        while len(points) < n_points:
            p = Point(random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            if poly.contains(p):                
                points.append((p.x, p.y))
        return points

def sample_point_in_multiple_polygon(polys, n_points=None):
    X_MIN, Y_MIN, X_MAX, Y_MAX = inf, inf, -inf, -inf
    for poly in polys:
        x_min, y_min, x_max, y_max = poly.bounds
        if x_min < X_MIN: X_MIN = x_min
        if y_min < Y_MIN: Y_MIN = y_min
        if x_max > X_MAX: X_MAX = x_max
        if y_max > Y_MAX: Y_MAX = y_max
   
    if n_points is None:
        while True:
            p = Point(random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX))
            for poly in polys:
                if poly.contains(p):
                        return (p.x, p.y)
    else:
        points = []
        while len(points) < n_points:
            p = Point(random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX))
            for poly in polys:
                if poly.contains(p):                
                    points.append((p.x, p.y))
        return points


## === math ===
def calculate_square_IOU(side, centerA, centerB):
    """Only for square submaps with the same side length.
    """
    dx = abs(centerB[0] - centerA[0])
    dy = abs(centerB[1] - centerA[1])
    if (dx >= side) or (dy >= side):
        return 0.0
    else:
        intersection = (side - dx) * (side - dy)
        union = 2 * (side**2) - intersection
        return intersection/union

calculate_submap_IOU = calculate_square_IOU

def calculate_circle_IOU(r, centerA, centerB):
    """Only for circle submaps with the same radius.
    """
    if type(centerA) is not np.ndarray:
        centerA = np.array(centerA)
    if type(centerB) is not np.ndarray:
        centerB = np.array(centerB)

    d = np.linalg.norm(centerA - centerB)
    if d >= 2 * r:
        return 0
    
    r2, d2 = r**2, d**2
    intersection = r2 * np.arccos((d2/2-r2)/r2) - d * np.sqrt(r2 - d2/4)
    union = 2 * np.pi * r2 - intersection
    return intersection/union

## === data collection ===
def collect_beams(pings):
    """Collect point cloud as np.array from 
    auvlib.data_tools.std_data.mbes_ping (beams).
    """
    pings_beams_list = list(map(lambda x: x.beams, pings))
    beams_list = list(itertools.chain(*pings_beams_list)) 
    beams_arr = np.array(beams_list)
    return beams_arr
    
def collect_trajectory(pings):
    """Collect point cloud as np.array from 
    auvlib.data_tools.std_data.mbes_ping (trajectory).  
    """
    trajectory_list = list(map(lambda x: x.pos_, pings))
    traj_arr = np.array(trajectory_list)
    return traj_arr

def sample_negative_pairs(pair_dataset, num_positive=None):
    """Randomly sample negative pairs from the test set.
    """
    if not num_positive:
        num_positive = len(pair_dataset)
    diameter = RADIUS["submap"] * 2
    random.seed(1)
    num_sampled = 0
    pair_list = []
    all_pairs = safe_load_pickle(f'{pair_dataset.processed_dir}/new_pairs.pkl')
    train_indices = list(set(list(map(lambda x: x[0], all_pairs))))
    test_indices = list(set(list(map(lambda x: x[1], all_pairs))))
    n_train, n_test = len(train_indices), len(test_indices)

    while num_sampled < num_positive:
        print(f"Counting down ... {num_sampled}")
        i, j = random.randint(0, n_train-1), random.randint(0, n_test-1)
        train_idx, test_idx = train_indices[i], test_indices[j]
        _, _, train_test_pose, _ = pair_dataset.get_train_test(train_idx, test_idx)
        train_test_pose = to_np(train_test_pose.pos).ravel().astype(np.float64)
        train_pose, test_pose = train_test_pose[:3], train_test_pose[3:]
        # if this pair has been collected
        if (train_idx, test_idx) in pair_list:
            continue
        # check submap distance
        if nl.norm(train_pose[:-1] - test_pose[:-1]) > diameter:
            # append if it is a negative pair
            pair_list.append((train_idx, test_idx))
            num_sampled += 1

    return pair_list
