import cv2
import matplotlib.pyplot as plt
import torch

from dataset import PairDatasetTest
from models import *
from param import *
from utils.dataset import sample_negative_pairs
from utils.feature_matching import (draw_matches_with_label,
                                    filter_match_by_depth, match_images)
from utils.helpers import integer_histogram
from utils.visualization import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

test_dataset_name = 'Circle100Test'
test_path = f'{DATA_PATH}{test_dataset_name}'
test_pair_positive = PairDatasetTest(root=test_path)

# ==== 1 sample negative pairs and save ====
# Do this only once
# negative_pair_list = sample_negative_pairs(test_pair_positive)
# safe_save_pickle(negative_pair_list, f'{test_pair.processed_dir}/negative_pairs.pkl')

test_pair_negative = PairDatasetTest(root=test_path, pair_path="negative_pairs.pkl")
num_positive_pairs, num_negative_pairs = len(test_pair_positive), len(test_pair_negative)
print(f"#positive pairs = {num_positive_pairs}, #negative pairs = {num_negative_pairs}")

model = MatcherTest(keypoint_thresh=0.7).to(device)
model_dict = torch.load(MODEL_PATH)
model.load_model(model_dict)
assert model.training == False

# parameters
model.keypoint_thresh = KEYPOINT_THRESHOLD
dist_thresh = MATCHING_THRESHOLD
PLOT_FEATURE_MATCHING = False
corr_positive, corr_negative = {}, {}

# ==== 2 test positive pairs ====
num_match_positive = []

for data_idx, data in enumerate(test_pair_positive):
    
    for item in data:
        item = item.to(device)
    data1, data2 = data[0], data[1]
    output = model(data1.clone(), data2.clone())

    if len(output) == 1:
        # False negative
        num_match = 0
        # false_negative += 1
        num_match_positive.append(num_match)
        continue
    else:
        assert len(output) == 6
        data_points, data_keypoints, feat_anc, feat_pos, abs1, abs2 = output

    img1, img2, depth_all = match_images(data_points, data_keypoints)

    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(feat_anc, feat_pos)
    sorted_matches = sorted(matches, key=lambda x:x.distance) 
    good_matches = list(filter(lambda x: True if x.distance < dist_thresh else False, sorted_matches))
    matches_used = filter_match_by_depth(data_keypoints, good_matches, threshold=DEPTH_FILTER_DIST)
    # possible to be TP or FN
    num_match = len(matches_used)
    num_match_positive.append(num_match)

    if PLOT_FEATURE_MATCHING and (num_match > 0):
        img_match, error = draw_matches_with_label(img1[0], img1[1], img2[0], img2[1], \
            matches=matches_used, label=(abs1, abs2), thresh=CORRECT_MATCH_DIST)
        print(f"data1 #keypoints: {data_keypoints['anc'].shape[0]}")
        print(f"data2 #keypoints: {data_keypoints['pos'].shape[0]}")
        mean_error = sum(error)/len(error)
        print(f"mean error = {mean_error}")
        plt.imshow(img_match)
        plt.title(f"mean error of {num_match} pairs (Unit: m) = {mean_error}")
        plt.show()

fig_pos = integer_histogram(num_match_positive)
plt.xlabel("#feature correspondences")
plt.ylabel("#submaps")
fig_pos.show()

# ==== 3 test negative pairs ====
num_match_negative = []
for data_idx, data in enumerate(test_pair_negative):
    
    for item in data:
        item = item.to(device)
    data1, data2 = data[0], data[1]
    output = model(data1.clone(), data2.clone())

    if len(output) == 1:
        # True negative
        num_match = 0
        num_match_negative.append(num_match)
        continue
    else:
        assert len(output) == 6
        data_points, data_keypoints, feat_anc, feat_pos, abs1, abs2 = output

    img1, img2, depth_all = match_images(data_points, data_keypoints)

    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(feat_anc, feat_pos)
    sorted_matches = sorted(matches, key=lambda x:x.distance) 
    good_matches = list(filter(lambda x: True if x.distance < dist_thresh else False, sorted_matches))
    matches_used = filter_match_by_depth(data_keypoints, good_matches, threshold=DEPTH_FILTER_DIST)
    # possible to be TP or FN
    num_match = len(matches_used)
    num_match_negative.append(num_match)

    if PLOT_FEATURE_MATCHING and (num_match > 0):
        img_match, error = draw_matches_with_label(img1[0], img1[1], img2[0], img2[1], \
            matches=matches_used, label=(abs1, abs2), thresh=CORRECT_MATCH_DIST)

        print(f"data1 #keypoints: {data_keypoints['anc'].shape[0]}")
        print(f"data2 #keypoints: {data_keypoints['pos'].shape[0]}")
        mean_error = sum(error)/len(error)
        print(f"mean error = {mean_error}")
        plt.imshow(img_match)
        plt.title(f"#correspondence = {num_match} ")
        plt.show()

fig_neg = integer_histogram(num_match_negative)
plt.xlabel("#feature correspondences")
plt.ylabel("#submaps")
fig_neg.show()
