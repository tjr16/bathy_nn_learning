"""This script contains visualization tools for point clouds.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from .data import numpy_to_pcd, to_np


def visualize_cloud(pcd, with_frame=False):
    """Visualize 3D point cloud with open3d.

    Args:
        pcd (open3d.cuda.pybind.geometry.PointCloud): from `o3d.geometry.PointCloud()`
        with_frame (bool, optional): Plot with reference frame in the center. It works
            only if `pcd` is a single point cloud. Defaults to False.
    """
    if type(pcd) is list:
        for idx, cloud in enumerate(pcd):
            if type(cloud) is np.ndarray:
                pcd[idx] = numpy_to_pcd(cloud)
        
        o3d.visualization.draw_geometries(pcd)

    else:
        if type(pcd) is np.ndarray:
            pcd = numpy_to_pcd(pcd)
            if with_frame:
                pcd_np = np.asarray(pcd.points)
                origin = pcd_np.mean(axis=0)
                size = pcd_np.ptp(axis=0).mean()
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin) 
                o3d.visualization.draw_geometries([pcd, frame])
            else:
                o3d.visualization.draw_geometries([pcd])

def visualize_cloud_keypoint(pcd):
    """Visualize point cloud with keypoints in black color.

    Args:
        pcd (open3d.cuda.pybind.geometry.PointCloud)
    """
    assert type(pcd) is list
    if type(pcd[0]) is np.ndarray:
        points = numpy_to_pcd(pcd[0])
    else:
        points = pcd[0]
    if type(pcd[1]) is np.ndarray:
        keypoints = numpy_to_pcd(pcd[1])
    else:
        keypoints = pcd[1]

    keypoints = o3d.geometry.PointCloud(keypoints)
    n_keypoints = np.array(keypoints.points).shape[0]
    colors = [[0, 0, 0] for i in range(n_keypoints)]  # black
    keypoints.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([points, keypoints])

def visualize_position(pcd_np, color=False, bounds=None):
    """Visualize (x, y) coordinates.

    Args:
        pcd_np (numpy.ndarray) 
        color (bool, optional): Use depth for color. Defaults to False.
        bounds (optional)
    """
    if color:
        plt.scatter(pcd_np[:, 0], pcd_np[:, 1], c=pcd_np[:, 2])
    else:
        plt.scatter(pcd_np[:, 0], pcd_np[:, 1])
    if bounds:
        plt.plot(bounds[0], bounds[1])   
    plt.grid()
    plt.axis("equal")
    plt.show()

def plot3d(xyz):
    """Plot point cloud in 3D figure.
    """
    ax = plt.axes(projection='3d')
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2])
    plt.show()

def pcd_depth_map(pcd, plot=True):
    """Project the depth map of a point cloud with Matplotlib

    Args:
        pcd (o3d.cuda.pybind.geometry.PointCloud or list)
        plot (bool, optional): If True, plot the image. If False, return the image. 
            Defaults to True.
    """
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible = False)
    if type(pcd) is list:
        for p in pcd:
            visualizer.add_geometry(p)
    else:
        visualizer.add_geometry(pcd)
    img = visualizer.capture_screen_float_buffer(True)
    if plot:
        plt.imshow(np.asarray(img))
        plt.show()
    else:
        return img

def plot_keypoints(data1, data2, idx1, idx2, weights = None, save_path=None):
    """Plot keypoints in 3D figures.
    """
    pos1 = to_np(data1.pos[idx1])
    pos2 = to_np(data2.pos[idx2])
    pos_all1 = to_np(data1.pos)
    pos_all2 = to_np(data2.pos)
    fig = plt.figure()#figsize=plt.figaspect(0.5))

    c3, c4 = pos_all1[:, 2].copy(), pos_all2[:, 2].copy()
    if weights:
        c1, c2 = weights[0], weights[1]
        c5, c6 = c3.copy(), c4.copy()
        c5[to_np(idx1)] = np.full(len(idx1), c3.max()+1)
        c6[to_np(idx2)] = np.full(len(idx2), c4.max()+1)
    else:
        c1, c2 = pos1[:, 2], pos2[:, 2]

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_title("Submap1, keypoints")
    ax1.scatter3D(pos1[:, 0], pos1[:, 1], pos1[:, 2], c=c1)

    ax2 = fig.add_subplot(2, 3, 4, projection='3d')
    ax2.set_title("Submap2, keypoints")
    ax2.scatter3D(pos2[:, 0], pos2[:, 1], pos2[:, 2], c=c2)

    ax3 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3.set_title("Submap1")
    ax3.scatter3D(pos_all1[:, 0], pos_all1[:, 1], pos_all1[:, 2], c=c3)

    ax4 = fig.add_subplot(2, 3, 5, projection='3d')
    ax4.set_title("Submap2")
    ax4.scatter3D(pos_all2[:, 0], pos_all2[:, 1], pos_all2[:, 2], c=c4)

    if weights:
        ax5 = fig.add_subplot(2, 3, 3, projection='3d')
        ax5.set_title("Submap1 with keypoints")
        ax5.scatter3D(pos_all1[:, 0], pos_all1[:, 1], pos_all1[:, 2], c=c5)

        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.set_title("Submap2 with keypoints")
        ax6.scatter3D(pos_all2[:, 0], pos_all2[:, 1], pos_all2[:, 2], c=c6)
    
    if save_path:
        with open(save_path,'wb') as fig_file:
            pickle.dump(fig, fig_file)
    plt.show()

