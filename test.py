import cv2
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from dataset import PairDatasetTest
from models import *
from param import *
from utils.data import numpy_to_pcd, save_pcd, to_np
from utils.feature_matching import (draw_matches_with_label,
                                    filter_match_by_depth, match_images)
from utils.helpers import Timer
from utils.visualization import *

SAVE_PATH = "pcd"
PLOT_FEATURE_MATCHING = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

test_dataset_name = 'Circle100Test'
test_path = f'{DATA_PATH}{test_dataset_name}'
test_pair = PairDatasetTest(test_path)
loader = DataLoader(test_pair, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)

print("In test set")
print(f"Data amount: {len(loader)}")

model = MatcherTest().to(device)
model_dict = torch.load(MODEL_PATH)
model.load_model(model_dict)
assert model.training == False

# parameters
model.keypoint_thresh = KEYPOINT_THRESHOLD
dist_thresh = MATCHING_THRESHOLD

if SAVE_PATH: save_id = 0

timer = Timer()
for data_idx, data in enumerate(loader):
    
    for item in data:
        item = item.to(device)

    data1, data2 = data[0], data[1]
    data1_clone, data2_clone = data1.clone(), data2.clone()
    timer.tic()
    output = model(data1_clone, data2_clone)
    timer.toc()
    
    if len(output) == 1:
        data_points = output
        print("No keypoints in one of the data")
        continue
    else:
        assert len(output) == 6
        data_points, data_keypoints, feat_anc, feat_pos, abs1, abs2 = output
    
    if SAVE_PATH:   
        # original point clouds with abs positions
        original1, original2 = to_np(data1.pos[:, 3:].clone()), to_np(data2.pos[:, 3:].clone())
        # moved original point clouds: original1_mean becomes the origin
        original1, original2 = original1.astype(np.float64), original2.astype(np.float64)
        # convert to float64 is because of numerical precision
        original1_mean = original1.mean(axis=0)
        original1_moved = original1 - original1_mean
        original2_moved = original2 - original1_mean
        # centered point clouds
        mean_depth = data_points['anc'][:, -1].mean()
        save_points1 = data_points['anc'].copy()
        save_points1[:, -1] = save_points1[:, -1] - mean_depth
        save_points2 = data_points['pos'].copy()
        save_points2[:, -1] = save_points2[:, -1] - mean_depth
        save_keypoints1 = data_keypoints['anc'].copy()
        save_keypoints1[:, -1] = abs1[:, -1] - mean_depth
        save_keypoints2 = data_keypoints['pos'].copy()
        save_keypoints2[:, -1] = abs2[:, -1] - mean_depth
        # convert to np.float64
        save_points1, save_points2 = save_points1.astype(np.float64), save_points2.astype(np.float64)
        save_keypoints1, save_keypoints2 = save_keypoints1.astype(np.float64), save_keypoints2.astype(np.float64)

    img1, img2, depth_all = match_images(data_points, data_keypoints)

    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(feat_anc, feat_pos)
    sorted_matches = sorted(matches, key=lambda x:x.distance)
    good_matches = list(filter(lambda x: True if x.distance < dist_thresh else False, sorted_matches))
    good_matches = filter_match_by_depth(data_keypoints, good_matches, threshold=DEPTH_FILTER_DIST)
    if len(good_matches) < MATCH_NUM_THRESHOLD:
        continue  # abondon this pair
    if len(good_matches) > 20:
        # take topK matches if there are too many matches
        matches_used = good_matches[:20]
    else:
        matches_used = good_matches

    # error: absolute position distance of these pairs
    img_match, error = draw_matches_with_label(img1[0], img1[1], img2[0], img2[1], \
        matches=matches_used, label=(abs1, abs2), thresh=CORRECT_MATCH_DIST)
    
    # successfully matched: save keypoints and points
    if SAVE_PATH:
        # original point cloud
        print(f"Save point cloud id={save_id}...")
        save_pcd(numpy_to_pcd(original1), f"{SAVE_PATH}/original1_{save_id}.pcd")
        save_pcd(numpy_to_pcd(original2), f"{SAVE_PATH}/original2_{save_id}.pcd")
        save_pcd(numpy_to_pcd(original1_moved), f"{SAVE_PATH}/original1_moved_{save_id}.pcd")
        save_pcd(numpy_to_pcd(original2_moved), f"{SAVE_PATH}/original2_moved_{save_id}.pcd")
        # centered point cloud; keypoints
        save_pcd(numpy_to_pcd(save_points1), f"{SAVE_PATH}/points1_{save_id}.pcd")
        save_pcd(numpy_to_pcd(save_points2), f"{SAVE_PATH}/points2_{save_id}.pcd")
        save_pcd(numpy_to_pcd(save_keypoints1), f"{SAVE_PATH}/keypoints1_{save_id}.pcd")
        save_pcd(numpy_to_pcd(save_keypoints2), f"{SAVE_PATH}/keypoints2_{save_id}.pcd")
        # save correspondences
        query_list, train_list = [], []
        for match in matches_used:
            queryIdx = match.queryIdx  # index_query
            trainIdx = match.trainIdx  # index_match
            query_list.append(queryIdx)
            train_list.append(trainIdx)
        query_str = ' '.join(map(str, query_list))
        match_str = ' '.join(map(str, train_list))
        with open(f"{SAVE_PATH}/query_idx_{save_id}.txt", "w") as f:
            f.write(query_str)
        with open(f"{SAVE_PATH}/match_idx_{save_id}.txt", "w") as f:
            f.write(match_str)
        save_id += 1

    correct_match = list(filter(lambda x: x < CORRECT_MATCH_DIST, error))

    if PLOT_FEATURE_MATCHING:
        print(f"data1 #keypoints: {data_keypoints['anc'].shape[0]}")
        print(f"data2 #keypoints: {data_keypoints['pos'].shape[0]}")
        mean_error = sum(error)/len(error)
        print(f"mean error = {mean_error}")
        plt.imshow(img_match)
        num_matches = len(matches_used)
        plt.title(f"mean error of {num_matches} pairs (Unit: m) = {mean_error}")
        plt.show()
