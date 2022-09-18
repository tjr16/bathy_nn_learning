import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl


def filter_match_by_depth(data_keypoints, matches, threshold=2.0):
    filtered_matches = []
    for m in matches:
        query_idx = m.queryIdx
        train_idx = m.trainIdx
        query_depth = data_keypoints['anc'][query_idx][-1]
        train_depth = data_keypoints['pos'][train_idx][-1]
        diff_depth = abs(query_depth - train_depth)
        if diff_depth <= threshold:
            filtered_matches.append(m)
    return tuple(filtered_matches)

def read_image(data_points, data_keypoints, name='anc'):
    SIZE_RATIO = 1.5
    points = data_points[name]
    points_xy = points[:, :-1]
    depth = points[:, 2]  # do NOT scale depth here

    size_image = np.ceil(points_xy.ptp()) * SIZE_RATIO  # points_xy should have been centered
    SIZE = int(size_image)
    middle = (points_xy.max() + points_xy.min())/2
    points_scaled = (SIZE - 1)/2 + (points_xy-middle)/points_xy.ptp() * (SIZE - 1)
    points_scaled = points_scaled.astype(int)
    rows = points_scaled[:, 1]
    cols = points_scaled[:, 0]

    keypoints = data_keypoints[name][:, :-1]
    # key_middle = (keypoints.max() + keypoints.min())/2
    keypoints_scaled = (SIZE - 1)/2 + (keypoints-middle)/points_xy.ptp() * (SIZE - 1)
    # keypoints_scaled = (SIZE - 1)/2 + (keypoints-key_middle)/keypoints.ptp() * (SIZE - 1)
    keypoints_scaled = keypoints_scaled.astype(int)
    key_rows = keypoints_scaled[:, 1]
    key_cols = keypoints_scaled[:, 0]
    return points_scaled, rows, cols, depth, SIZE, keypoints_scaled, key_rows, key_cols

def color_image(rows, cols, depth, SIZE):
    img = np.ones((SIZE, SIZE)) * 122  #(255-1)/2. Set mean value for color map only.
    img = img.astype(np.uint8)
    img[rows, cols] = depth
    img = cv2.applyColorMap(img,cv2.COLORMAP_JET)

    img_color = np.ones((SIZE, SIZE, 3)) * 255  # white
    img_color = img_color.astype(np.uint8)
    img_color[rows, cols] = img[rows, cols]
    return img_color

def match_images(data_points, data_keypoints, name1='anc', name2='pos'):
    """
    Args:
        data_points (dict): points of the two images, each element: (num_points, 3)
        data_keypoints (dict): keypoints of the two images, each element: (num_clusters, 3)
        name1 (str): name of the first image
        name2 (str): name of the second image
    Returns:
        img_color1 (np.array): color image of the first image
        img_color2 (np.array): color image of the second image
        depth_all (np.array): absolute depth of all points
    """
    points1, rows1, cols1, depth1, size1, keypoints1, key_rows1, key_cols1 = \
        read_image(data_points, data_keypoints, name1)
    points2, rows2, cols2, depth2, size2, keypoints2, key_rows2, key_cols2 = \
        read_image(data_points, data_keypoints, name2)

    depth_all = np.concatenate([depth1, depth2])
    depth1_scaled = (depth1 - depth_all.min())/ depth_all.ptp() * 255
    depth2_scaled = (depth2 - depth_all.min())/ depth_all.ptp() * 255

    SIZE = max(size1, size2)
    img_color1 = color_image(rows1, cols1, depth1_scaled, SIZE)
    img_color2 = color_image(rows2, cols2, depth2_scaled, SIZE)

    # img_new[rows_key_anc, cols_key_anc] = (0, 0, 0)  # black
    cv_keypoints1 = [cv2.KeyPoint(keypoints1[i, 0].astype(np.float64), keypoints1[i, 1].astype(np.float64), 1.0)
                        for i in range(keypoints1.shape[0])]
    cv_keypoints2 = [cv2.KeyPoint(keypoints2[i, 0].astype(np.float64), keypoints2[i, 1].astype(np.float64), 1.0)
                        for i in range(keypoints2.shape[0])]
    return (img_color1, cv_keypoints1), (img_color2, cv_keypoints2), depth_all

def plot_match_distance(match):
    """Plot distance histogram of matched features.

    Args:
        match (tuple[cv2.DMatch]): matches
    """
    dist_list = np.array(list(map(lambda x: x.distance, match)))
    plt.hist(dist_list, bins=30)
    plt.title("histogram: distance between salient keypoint pairs")
    plt.show()

def draw_matches_with_label(img1, kp1, img2, kp2, matches, label, thresh=10):
    """Draw feature matching.
    """
    RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    CORRECT_MATCH = GREEN
    WRONG_MATCH = RED
    CIRCLE = BLUE

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    img_out = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
    # first image: left
    img_out[:rows1, :cols1, :] = img1
    # second image: right
    img_out[:rows2, cols1:cols1+cols2, :] = img2

    distance_list = []
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        # x: col, y: row
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # draw circles
        r = 4
        cv2.circle(img=img_out, center=(int(x1), int(y1)), radius=r, color=CIRCLE, thickness=1)
        cv2.circle(img_out, (int(x2)+cols1, int(y2)), r, CIRCLE, 1)
        # draw lines
        xy1, xy2 = label[0][img1_idx][:-1], label[1][img2_idx][:-1]
        distance = nl.norm(xy1 - xy2)
        if distance < thresh:
            cv2.line(img=img_out, pt1=(int(x1),int(y1)), pt2=(int(x2)+cols1,int(y2)), color=CORRECT_MATCH, thickness=1)
        else:
            cv2.line(img_out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), WRONG_MATCH, 1)
        distance_list.append(distance)
    return img_out, distance_list
