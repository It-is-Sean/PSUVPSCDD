# Copyright (c) 2026 Weirong Chen
import os
import numpy as np

import open3d as o3d




# create a function to save ply with outlier removal using open3d, given a save filename, a numpy array of shape Nx3 for point cloud, and a numpy array of shape Nx3 for colors (optional)
def save_point_cloud_with_outlier_removal(filename, xyz, rgb=None, remove_outliers=True):
    """
    Save a point cloud with optional outlier removal using Open3D.
    Args:
        xyz (np.ndarray): Nx3 array of 3D points.
        rgb (np.ndarray, optional): Nx3 array of RGB colors for each point.
        filename (str): Path to save the PLY file.
        remove_outliers (bool): Whether to remove outliers using statistical outlier removal.
    """
    if filename is None:
        raise ValueError("filename must be provided")

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # If colors are provided, set them
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # Normalize to [0,1]

    # Remove outliers if requested
    if remove_outliers:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Removed {len(xyz) - len(ind)} outliers")

    # Save the point cloud
    o3d.io.write_point_cloud(filename, pcd)
