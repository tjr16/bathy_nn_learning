import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Softplus
from torch_geometric.nn import (MLP, Linear, PointNetConv, fps,
                                global_max_pool, knn_interpolate, radius)

from param import *
from utils.data import to_np


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=CONFIG["SetAbstraction"]["max_num_neighbors"])
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class Descriptor(torch.nn.Module):
    def __init__(self, r, nn):
        super().__init__()
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch, key_idx):
        assert batch.max() == batch[key_idx].max()
        row, col = radius(pos, pos[key_idx], self.r, batch, batch[key_idx],
                        max_num_neighbors=CONFIG["Descriptor"]["max_num_neighbors"])
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[key_idx]
        x = self.conv((x, x_dst), (pos, pos[key_idx]), edge_index)
        pos, batch = pos[key_idx], batch[key_idx]
        x = F.normalize(x, dim=-1)
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2(torch.nn.Module):
    def __init__(self, pos_dim=3, feat_dim=1):
        super().__init__()

        self.position_dim = pos_dim
        self.feature_dim = feat_dim
        self.out_dim = CONFIG["PointNet2"]["dim_output"]

        self.sa1_module = SAModule(ratio=0.2, r=CONFIG["PointNet2"]["r1"], nn=MLP([self.feature_dim + self.position_dim, 32, 32]))
        self.sa2_module = SAModule(ratio=0.25, r=CONFIG["PointNet2"]["r2"], nn=MLP([32 + self.position_dim, 32, 64]))
        self.sa3_module = GlobalSAModule(MLP([64 + self.position_dim, 64, 64]))

        self.fp3_module = FPModule(1, MLP([64 + 64, 64, 64]))
        self.fp2_module = FPModule(3, MLP([64 + 32, 32, 32]))
        self.fp1_module = FPModule(3, MLP([32 + self.feature_dim, 32, 32, 32]))

        self.mlp = MLP([32, 32, 32, self.out_dim], dropout=0.5, batch_norm=False)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        return self.mlp(x)


class Detector(torch.nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.k = k
        self.k_pos = int(SIZE_SUBMAP * CONFIG["Detector"]["ratio"])  # position filtering
        self.mlp = Sequential(
            MLP([32, 16, 8]),  # default: dropout=0, batch_norm=True
            Linear(8, 1),
            Softplus()
        )
        # self.use_detector = True
        self.R = RADIUS["submap"]
        self.r = RADIUS["patch"]
    
    # def use_detector(self, use: bool = True):
    #     self.use_detector = use

    def forward(self, x, pos, batch):
        # dense feature, position, batch
        batch_size = batch.max().item() + 1
        x = self.mlp(x)
        x_batch = x.reshape(batch_size, -1)
        pos_batch = pos.reshape(batch_size, -1, 3)
        pos_xy_batch = pos_batch[:, :, :-1]
        pos_dist_center = pos_xy_batch.norm(dim=-1)

        _, indices1 = torch.topk(-pos_dist_center, k=self.k_pos)  # 4 X 6553

        pos_batch_selected = torch.gather(pos_batch, dim=1, index=indices1.unsqueeze(-1).repeat(1, 1, 3))
        batch_selected = torch.repeat_interleave(torch.arange(batch_size), self.k_pos).to(pos_batch_selected.device)
        pos_selected = pos_batch_selected.reshape(-1, 3)
        
        key_idx = fps(pos_selected, batch_selected, ratio=self.k/self.k_pos)
        key_idx_batch = key_idx.reshape(batch_size, -1)
        key_idx_batch = key_idx_batch \
            - torch.arange(0, batch_size*self.k_pos, self.k_pos).reshape(-1, 1).to(key_idx_batch.device)

        original_indices = torch.gather(indices1, dim=1, index=key_idx_batch)
        weights = torch.gather(x_batch, dim=1, index=original_indices)

        return weights, original_indices


class Matcher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense_feat = PointNet2(pos_dim=3, feat_dim=1)
        self.feature_dim = self.dense_feat.out_dim
        self.desc_dim = CONFIG["Descriptor"]["dim_output"]
        self.detector = Detector(k=512)
        self.descriptor = Descriptor(r=RADIUS["patch"], \
            nn=MLP([self.dense_feat.out_dim + self.dense_feat.position_dim, 32, self.desc_dim]))

    def forward(self, data1, data2, center):
        assert data1.pos.shape[1] == 6
        abs_pos1, abs_pos2 = data1.pos[:, 3:], data2.pos[:, 3:]      # absolute pos
        data1.pos, data2.pos = data1.pos[:, :3], data2.pos[:, :3]    # centered pos
        
        data1.x = torch.sin(CONFIG["Matcher"]["depth_period"] * data1.pos[:, -2:-1])
        data2.x = torch.sin(CONFIG["Matcher"]["depth_period"] * data2.pos[:, -2:-1])
        # relative position
        # data1.pos[:, -1] = data1.pos[:, -1] - data1.pos[:, -1].mean()
        # data2.pos[:, -1] = data2.pos[:, -1] - data2.pos[:, -1].mean()
 
        batch_size = data1.batch.max().item() + 1

        dense_src = self.dense_feat(data1)
        dense_tgt = self.dense_feat(data2)

        key_weights_batch, key_indices_batch = self.detector(dense_src, data1.pos, data1.batch)  # (4, 64) batched weights
        device0 = key_indices_batch.device
        key_indices1 = torch.arange(start=0, end=data1.batch.numel(), step=SIZE_SUBMAP).reshape(-1, 1).to(device0)
        key_indices = (key_indices_batch + key_indices1).ravel()
        key_weights = key_weights_batch.ravel()  # all in one vector

        # ========== triplets ==========
        center_xy1 = center.pos[:, :2].double()  # 01
        center_xy2 = center.pos[:, 3:-1].double()  # 34

        # all absolute position
        xyz1_batch = abs_pos1.reshape(batch_size, -1, 3)
        xyz2_batch = abs_pos2.reshape(batch_size, -1, 3)

        # key absolute position
        key_xyz1_batch = torch.gather(xyz1_batch, dim=1, index=key_indices_batch.unsqueeze(-1).repeat(1, 1, 3))
        key_xy1_batch = key_xyz1_batch[:, :, :-1].double()
        dr = self.detector.R - self.descriptor.r  # distance to both centers should be less than dr

        # XY distance: keypoints in submap1 to center1
        dist_key1_center1 = torch.cdist(key_xy1_batch, center_xy1.unsqueeze(1)).squeeze(-1)
        dist_key1_center2 = torch.cdist(key_xy1_batch, center_xy2.unsqueeze(1)).squeeze(-1)
        # overlap1_maski: points in submap1: in submapi or not?
        # overlap1_mask: points in submap1: in XY overlap region
        overlap1_mask1 = dist_key1_center1 < dr
        overlap1_mask2 = dist_key1_center2 < dr
        overlap1_mask = torch.logical_and(overlap1_mask1, overlap1_mask2)

        # distance matrix
        # NOTE: must convert it to double before using torch.cdist
        dist_mat = torch.cdist(key_xyz1_batch.double(), xyz2_batch.double())  # use closest point in 3D space, bS X nKey X nPoints
        min_dist = torch.min(dist_mat, dim=2)
        min_distances_batch, min_indices_batch = min_dist.values, min_dist.indices  # batchSize X nKey
        min_indices1 = torch.arange(start=0, end=data1.batch.numel(), step=SIZE_SUBMAP).reshape(-1, 1).to(device0)
        min_indices = (min_indices_batch + min_indices1).ravel()

        THRESHOLD = 1
        distance_mask = min_distances_batch < THRESHOLD  # distance mask

        # good mask for keypoints in submap1
        anchor_mask_batch = torch.logical_and(overlap1_mask, distance_mask) # bS X nP
        anchor_mask = anchor_mask_batch.ravel()

        anchor_key_idx_batch = anchor_mask_batch.nonzero(as_tuple=True)
        anchor_key_idx = anchor_mask.nonzero(as_tuple=True)  # not original index
        anchor_idx = key_indices[anchor_key_idx]  # original indices among nPoints points
        anchor_weights = key_weights[anchor_key_idx]

        # Exception
        if anchor_idx.shape[0] == 0: return 0  # no triplets in this pair, no loss to train

        positive_idx = min_indices[anchor_key_idx]

        # === select negative ===
        # select negative from TOP 85% points far away from anchor
        k_dist = int(SIZE_SUBMAP * 0.85)
        n_negative = 5  # how many negative patches to sample
        dist_values, dist_indices = torch.topk(dist_mat, k_dist)  # dist_indices: original idx
        
        negative_sample_indices = torch.randint(
            low=0, high=k_dist, size=(batch_size, self.detector.k, n_negative)
            ).to(dist_indices.device)  # 4 X 64 X n_negative
        
        negative_indices_batch0 = torch.gather(dist_indices, dim=2, index=negative_sample_indices)


        anchor_indices = torch.repeat_interleave(anchor_idx, n_negative)
        anchor_weights_repeat = torch.repeat_interleave(anchor_weights, n_negative)
        positive_indices = torch.repeat_interleave(positive_idx, n_negative)

        negative_indices1 = torch.arange(start=0, end=data1.batch.numel(), step=SIZE_SUBMAP).reshape(-1, 1, 1).to(device0)
        negative_indices_batch = negative_indices_batch0 + negative_indices1

        negative_indices = negative_indices_batch[anchor_key_idx_batch[0], anchor_key_idx_batch[1], :].ravel()
        
        assert len(anchor_indices) == len(positive_indices) == len(negative_indices)

        # x_anchor, x_positive, x_negative: features retrieved by indices
                # local_features of source point cloud
        # # (batchSize * numKey, 32), (batchSize * numKey, 3), (batchSize * numKey)

        # NOTE: remove bad batches without triplets instead of skipping all batches
        # solve this issue: https://github.com/pyg-team/pytorch_geometric/issues/1615
        available_batch = torch.unique(data1.batch[anchor_indices])
        n_batch = len(available_batch)
        
        if n_batch < batch_size:
            all_batch = torch.arange(0, batch_size).to(available_batch.device)

            # mask for available batches
            mask0 = data1.batch == -1
            for batchId in available_batch:
                mask0 = torch.logical_or(mask0, data1.batch == batchId)

            # remove feature and position vectors in unavailable batches
            feat_src = dense_src[mask0, :]
            feat_tgt = dense_tgt[mask0, :]
            pos_src = data1.pos[mask0, :]
            pos_tgt = data2.pos[mask0, :]

            batch_diff = available_batch - all_batch[:len(available_batch)]
            for available_idx, batchId in enumerate(available_batch):
                low = (batchId * SIZE_SUBMAP).to(available_batch.device)
                high = ((1+batchId) * SIZE_SUBMAP).to(available_batch.device)
                idx_diff = batch_diff[available_idx] * SIZE_SUBMAP
                mask1 = torch.logical_and(low <= anchor_indices, anchor_indices < high)
                mask2 = torch.logical_and(low <= positive_indices, positive_indices < high)
                mask3 = torch.logical_and(low <= negative_indices, negative_indices < high)
                anchor_indices[mask1] = anchor_indices[mask1] - idx_diff
                positive_indices[mask2] = positive_indices[mask2] - idx_diff
                negative_indices[mask3] = negative_indices[mask3] - idx_diff
            
            batch_src = torch.repeat_interleave(all_batch[:len(available_batch)], SIZE_SUBMAP)
            batch_tgt = torch.repeat_interleave(all_batch[:len(available_batch)], SIZE_SUBMAP)

        else:
            feat_src, feat_tgt, pos_src, pos_tgt, batch_src, batch_tgt = \
                dense_src, dense_tgt, data1.pos, data2.pos, data1.batch, data2.batch

        x_anc, pos_anc, batch_anc = self.descriptor(feat_src, pos_src, batch_src, anchor_indices)
        x_pos, pos_pos, batch_pos = self.descriptor(feat_tgt, pos_tgt, batch_tgt, positive_indices)
        x_neg, pos_neg, batch_neg = self.descriptor(feat_tgt, pos_tgt, batch_tgt, negative_indices)

        # Normalized weights over batch
        anchor_weights_repeat = anchor_weights_repeat / anchor_weights_repeat.sum()

        return (x_anc, x_pos, x_neg), anchor_weights_repeat, \
            (anchor_indices, positive_indices, negative_indices, abs_pos1, abs_pos2)


class MatcherTest(Matcher):
    def __init__(self, keypoint_thresh=None):
        super().__init__()
        self._model_loaded = False
        self._keypoint_thresh = keypoint_thresh
    
    def load_model(self, model_obj):
        self.load_state_dict(model_obj['model_state_dict'])
        self.eval()
        self.extractor = self.dense_feat
        self.detector.k = self.detector.k_pos
        self._model_loaded = True
    
    @property
    def model_loaded(self):
        return self._model_loaded

    @property
    def keypoint_thresh(self):
        return self._keypoint_thresh

    @keypoint_thresh.setter
    def keypoint_thresh(self, value):
        self._keypoint_thresh = value

    @keypoint_thresh.deleter
    def keypoint_thresh(self):
        self._keypoint_thresh = None
          
    def forward(self, data1, data2, center=None):
        """
        Forward method for MatcherTest
        Only support batch_size == 1
        """
        assert self._model_loaded == True
        assert data1.pos.shape[1] == 6
        # If DataLoader is not used, should give batch indices here.
        if data1.batch is None:
            data1.batch = torch.zeros(data1.pos.shape[0]).long().to(data1.pos.device)
        if data2.batch is None:
            data2.batch = torch.zeros(data2.pos.shape[0]).long().to(data2.pos.device)

        abs_pos1, abs_pos2 = data1.pos[:, 3:].clone(), data2.pos[:, 3:].clone()      # absolute pos
        data1.pos, data2.pos = data1.pos[:, :3], data2.pos[:, :3]    # centered pos
        
        data1.x = torch.sin(CONFIG["Matcher"]["depth_period"] * data1.pos[:, -2:-1])
        data2.x = torch.sin(CONFIG["Matcher"]["depth_period"] * data2.pos[:, -2:-1]) 
        # relative position
        # data1.pos[:, -1] = data1.pos[:, -1] - data1.pos[:, -1].mean()
        # data2.pos[:, -1] = data2.pos[:, -1] - data2.pos[:, -1].mean()

        dense_src = self.extractor(data1)
        dense_tgt = self.extractor(data2)

        key_weights1, key_indices1 = self.detector(dense_src, data1.pos, data1.batch)
        key_weights2, key_indices2 = self.detector(dense_tgt, data2.pos, data2.batch)

        key_weights1 = key_weights1.ravel()
        key_weights2 = key_weights2.ravel()
        key_indices1 = key_indices1.ravel()
        key_indices2 = key_indices2.ravel()

        if self._keypoint_thresh:
            weight_thresh = self._keypoint_thresh
        else:
            DETECT_THRESH = 0.7
            # print('key_weights1:', torch.quantile(key_weights1, DETECT_THRESH).item())
            # print('key_weights2:', torch.quantile(key_weights2, DETECT_THRESH).item())
            weight_thresh = min(torch.quantile(key_weights1, DETECT_THRESH), \
                        torch.quantile(key_weights2, DETECT_THRESH))
        
        idx1 = key_indices1[key_weights1 > weight_thresh]
        idx2 = key_indices2[key_weights2 > weight_thresh]

        if self.keypoint_thresh:
            if (idx1.shape[0] == 0) or (idx2.shape[0] == 0):
                data_points = {}
                data_points['anc'] = data1.pos.cpu().numpy()
                data_points['pos'] = data2.pos.cpu().numpy()
                # original depth
                data_points['anc'][:, -1] = abs_pos1.cpu().numpy()[:, -1]
                data_points['pos'][:, -1] = abs_pos2.cpu().numpy()[:, -1]
                return [data_points] # no correspondence

        # absolute position of keypoints
        abs1 = to_np(abs_pos1[idx1, :])
        abs2 = to_np(abs_pos2[idx2, :])

        # relative position of keypoints
        pos1 = to_np(data1.pos[idx1])
        pos2 = to_np(data2.pos[idx2])
        # pos_all1 = to_np(data1.pos)
        # pos_all2 = to_np(data2.pos)

        feature1, _, _ = self.descriptor(dense_src, data1.pos, data1.batch, idx1)
        feature2, _, _ = self.descriptor(dense_tgt, data2.pos, data2.batch, idx2)

        feat1, feat2 = feature1.squeeze().detach().cpu().numpy(), feature2.squeeze().detach().cpu().numpy()
        if len(feat1.shape) == 1:
            feat1 = np.expand_dims(feat1, axis=0)
        if len(feat2.shape) == 1:
            feat2 = np.expand_dims(feat2, axis=0)

        score = {}  # not normalized
        score['anc'], score['pos'] = to_np(key_weights1).squeeze(), to_np(key_weights2).squeeze()

        data_keypoints, data_points = {}, {}
        data_keypoints['anc'] = pos1  # relative position of keypoints
        data_keypoints['pos'] = pos2
        data_points['anc'] = data1.pos.cpu().numpy()
        data_points['pos'] = data2.pos.cpu().numpy()
        # original depth
        data_points['anc'][:, -1] = abs_pos1.cpu().numpy()[:, -1]
        data_points['pos'][:, -1] = abs_pos2.cpu().numpy()[:, -1]

        # all relative points of all points, rela pos of keypoints,
        # feat vec of keypoints, abs pos of keypoints
        # get relative pos of keypoints: data_keypoints['anc'], data_keypoints['pos']
        return data_points, data_keypoints, feat1, feat2, abs1, abs2
