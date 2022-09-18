import math
import numbers
import os.path as osp
import pickle
import random
from itertools import repeat
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, HeteroData, InMemoryDataset
from torch_geometric.transforms import BaseTransform, LinearTransformation

from param import NUM_SUBMAPS
from utils.data import read_bin, read_pcd


class SubmapDataset(Dataset):

    def __init__(self, root, name="Circle100", transform=None):
        super().__init__(root)
        # DO NOT use properties: "transform" or "pre_transform"
        self.name = name
        self.num_submaps = NUM_SUBMAPS[self.name]
        if not transform:
            transform = SubmapTransform()
        self.submap_transform = transform
        # FIXME: submap_transform must not be None
        self.transform = None
        self.pre_transform = None

    def len(self):
        return self.num_submaps

    def get(self, idx):
        """
        Get numpy array from .bin files.
        NOTE: idx begins from 1, not 0.
        """
        pcd_np = read_bin(osp.join(self.processed_dir, f"processed_{idx}.bin"))
        pcd_data = Data(pos=torch.as_tensor(pcd_np))
        self.submap_transform(pcd_data)
        center = pcd_data.pos[0][3:]
        pcd_data.pos = pcd_data.pos[1:]
        return pcd_data, center


class PairDataset(InMemoryDataset):
    """Submap pairs in training and validation set.
    `train_idx` refers to the source point cloud,
    `test_idx` refers to the target point cloud.
    """

    def __init__(self, root, submap_set):
        super().__init__(root)  # do not use transform here
        self.submap_set = submap_set
        
        with open(osp.join(self.processed_dir, "positive_pair_list.pkl"), 'rb') as f:
            self.positive_pair_list = pickle.load(f)

    def len(self):
        return len(self.positive_pair_list)

    def get(self, idx):
        # index
        idx_src, idx_tgt = self.positive_pair_list[idx]
        # Data
        data_src, center_src = self.submap_set[idx_src]
        data_tgt, center_tgt = self.submap_set[idx_tgt]
        center_src = center_src.reshape((1, 3))
        center_tgt = center_tgt.reshape((1, 3))
        center_src_tgt = Data(pos=torch.cat((center_src, center_tgt), dim=1))        
        # output Data.pos.shape:
        # data_src, data_tgt: (batch_size * num_points), (3 relative xyz + 3 absolute xyz)
        # center_src_tgt: batch_size, (3 src center absolute xyz + 3 tgt center absolute xyz)
        return data_src, data_tgt, center_src_tgt


class PairDatasetTest(Dataset):
    """Submap pairs in tgt set.
    `train_idx.pcd` refers to the source point cloud,
    `test_idx.pcd` refers to the target point cloud.
    """

    def __init__(self, root, pair_path="new_pairs.pkl"):
        super().__init__(root)  # do not use transform here
        
        with open(osp.join(self.processed_dir, pair_path), 'rb') as f:
            self.pair_list = pickle.load(f)

    def len(self):
        return len(self.pair_list)

    def get_submap_with_src_tgt(self, src_idx, tgt_idx):
        src_pcd = read_pcd(self.processed_dir + f"/train_{src_idx}.pcd")
        tgt_pcd = read_pcd(self.processed_dir + f"/test_{tgt_idx}.pcd")
        src_tensor = torch.tensor(np.array(src_pcd.points), dtype=torch.float32)
        tgt_tensor = torch.tensor(np.array(tgt_pcd.points), dtype=torch.float32)
        # first point is the center of the submap, others are the points of the submap
        src_pose = src_tensor[0, :].reshape((1, 3))
        tgt_pose = tgt_tensor[0, :].reshape((1, 3))
        src_cloud = src_tensor[1:, :]
        tgt_cloud = tgt_tensor[1:, :]
        # Data
        src_cloud_centered = src_cloud.clone()
        tgt_cloud_centered = tgt_cloud.clone()
        assert src_cloud_centered.shape[-1] == 3
        src_cloud_centered[:, :-1] = src_cloud_centered[:, :-1] - src_pose[:, :-1]
        tgt_cloud_centered[:, :-1] = tgt_cloud_centered[:, :-1] - tgt_pose[:, :-1]

        src_tgt_pose = Data(pos=torch.cat((src_pose, tgt_pose), dim=1))  # 1X6
        data_src = Data(pos=torch.cat((src_cloud_centered, src_cloud), dim=1))
        data_tgt = Data(pos=torch.cat((tgt_cloud_centered, tgt_cloud), dim=1))
        indices = Data(x=torch.tensor([src_idx, tgt_idx]))
        
        return data_src, data_tgt, src_tgt_pose, indices

    def get(self, idx):
        # index in training set and test set (first pass and revisit)
        src_idx, tgt_idx = self.pair_list[idx]
        # print(src_idx, tgt_idx)
        return self.get_submap_with_src_tgt(src_idx, tgt_idx)
    
    def get_src_tgt_idx(self, idx):
        return self.pair_list[idx]


class SubmapTransform(BaseTransform):
    """Keep absolute position as labels.
    Center, rotate and randomly translate.
    """

    def __init__(self, degrees=(-3, 3), translate=0.2, noise=(0, 0, 0.05)) -> None:
        # degrees=(-180, 180)
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.translate = translate
        self.noise = noise

    def __call__(self, data: Union[Data, HeteroData]):
        assert hasattr(data.node_stores[0], 'pos')
        # Save a copy with absolute position
        # (N+1, 3), first element is the centroid
        data_clone = data.clone()
        
        # Step 1. center
        assert data.pos.size(-1) == 3
        data.pos[:, :-1] = data.pos[:, :-1] - data.pos[0, :-1]
        # Step 2. rotate
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        data = LinearTransformation(torch.tensor(matrix))(data)
        # Step 3. translate
        n, t = data.pos.size()[0], self.noise
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=3))
        ts = []
        for d in range(2):
            ts.append(data.pos.new_empty(n).normal_(mean=0, std=abs(t[d])))
        z_translate = random.uniform(-abs(self.translate), abs(self.translate))
        ts.append(data.pos.new_empty(n).normal_(mean=z_translate, std=abs(t[2])))
        data.pos = data.pos + torch.stack(ts, dim=-1)

        # Concatenate relative position and absolute position
        data.pos = torch.cat((data.pos, data_clone.pos), dim=-1)
        return data # (1+N, 6)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(rotate={self.degrees}, \
            translate={self.translate}, noise={self.noise})'
