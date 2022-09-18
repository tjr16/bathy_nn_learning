import os
import pickle
import re
import sys

import numpy as np
import open3d as o3d
import torch


## === data type conversion === 
def to_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

tensor_to_np = to_np

def numpy_to_pcd(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    return pcd

def pcd_to_numpy(pcd):
    pcd_np = np.asarray(pcd.points)
    return pcd_np

## === file saving and reading ===
def save_pcd(pcd, to_path):
    o3d.io.write_point_cloud(to_path, pcd)

def read_pcd(path):
    return o3d.io.read_point_cloud(path)

def save_bin(pcd_np: np.ndarray, to_path: str, dtype='float32'):
    pcd_np.astype(dtype).tofile(to_path)

def read_bin(file_name, dtype=None):
    """
    Read bin files and get xyz points
    """
    if not dtype:
        dtype=np.float32
    pcd_np = np.fromfile(file_name, dtype=dtype).reshape((-1, 3))
    return pcd_np

def safe_save_pickle(data, save_path, overwrite=False):
    if os.path.exists(save_path) and not overwrite:  # avoid overwriting
        print("File already exists. Set overwrite=True")
    else:
        with open(save_path, 'wb') as file:
            pickle.dump(data, file)

def safe_load_pickle(load_path):
    if not os.path.exists(load_path):
        print("File does not exist.")
        return None
    else:
        with open(load_path, 'rb') as file:
            return pickle.load(file)

def safe_save_pcd(pcd, save_path, overwrite=False):
    if os.path.exists(save_path) and not overwrite:  # avoid overwriting
        print("File already exists. Set overwrite=True")
    else:
        if type(pcd) is np.ndarray:
            pcd = numpy_to_pcd(pcd)
        o3d.io.write_point_cloud(save_path, pcd)

def pcd_header(n_points):
    """Define PCD header for customized descriptors.
    Note that the header is intended for size=4 and count=32.

    Args:
        n_points (int): Number of points in the point cloud.

    Returns:
        (str): PCD header.
    """
    # define pcd header for my own descriptor
    return \
    "# .PCD v0.7 - Point Cloud Data file format\n" + \
    "VERSION 0.7\n" + \
    "FIELDS nn_feat\n" + \
    "SIZE 4\n" + \
    "TYPE F\n" + \
    "COUNT 32\n" + \
    f"WIDTH {n_points}\n" + \
    f"HEIGHT 1\n" + \
    "VIEWPOINT 0 0 0 1 0 0 0\n" + \
    f"POINTS {n_points}\n" + \
    "DATA ascii\n"

def save_as_pcd_file(path, array):
    """Save customized descriptors as PCD file.
    """
    with open(path, "w") as f:
        f.write(pcd_header(array.shape[0]))
        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            array_str = np.array2string(array)
        array_str = re.sub('\[|\]', '', array_str)
        array_str = re.sub(' +', ' ', array_str)
        array_str = re.sub('\\n ', '\\n', array_str)
        array_str = array_str.strip()
        f.write(array_str)
