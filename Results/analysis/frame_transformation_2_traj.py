import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def display_map_with_points_map_frame(image, map_points, origin, resolution,
                                      estimated_points_pure, estimated_points_go_icp):
    """
    Displays the map image in the map (meter) coordinate frame,
    and plots the given map points and two estimated trajectories.

    Parameters:
    - image: 2D numpy array (map image, grayscale)
    - map_points: Nx2 numpy array of (x, y) points in meters (map frame)
    - origin: (x0, y0): bottom-left of map image in meters
    - resolution: meters per pixel
    - estimated_points_pure: Nx2 numpy array (trajectory using pure ICP)
    - estimated_points_go_icp: Nx2 numpy array (trajectory using Go-ICP)
    """
    x0, y0 = origin
    height, width = image.shape[:2]

    # Compute image bounds in map frame (meters)
    x1 = x0 + width * resolution
    y1 = y0 + height * resolution

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', extent=[x0, x1, y0, y1], origin='upper')

    num_points = map_points.shape[0]

    # True trajectory (control points)
    for i in range(num_points - 1):
        p1 = map_points[i]
        p2 = map_points[i + 1]
        dx, dy = p2 - p1
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', linewidth=1.5)
        arrow_start = p1 + 0.5 * np.array([dx, dy])
        plt.arrow(arrow_start[0], arrow_start[1], 0.2 * dx, 0.2 * dy,
                  head_width=0.15, head_length=0.2, fc='green', ec='green', length_includes_head=True)

    # Control points
    if num_points > 2:
        plt.scatter(map_points[1:-1, 0], map_points[1:-1, 1], c='green', s=30, label='Intermediate Points')
    plt.scatter(map_points[0, 0], map_points[0, 1], c='blue', s=50, label='Start Point')
    if num_points > 1:
        plt.scatter(map_points[-1, 0], map_points[-1, 1], c='red', s=30, label='End Point')

    # --- Pure ICP trajectory ---
    if estimated_points_pure is not None and len(estimated_points_pure) > 1:
        plt.plot(estimated_points_pure[:, 0], estimated_points_pure[:, 1],
                 linestyle='--', color='orange', linewidth=2, label='Estimated Trajectory: Pure ICP')
        plt.scatter(estimated_points_pure[0, 0], estimated_points_pure[0, 1],
                    marker='*', c='blue', s=80, label='Pure ICP Start Point')
        plt.scatter(estimated_points_pure[-1, 0], estimated_points_pure[-1, 1],
                    marker='*', c='red', s=80, label='Pure ICP End Point')

    # --- Go-ICP trajectory ---
    if estimated_points_go_icp is not None and len(estimated_points_go_icp) > 1:
        plt.plot(estimated_points_go_icp[:, 0], estimated_points_go_icp[:, 1],
                 linestyle='--', color='purple', linewidth=2, label='Estimated Trajectory: Go-ICP')
        plt.scatter(estimated_points_go_icp[0, 0], estimated_points_go_icp[0, 1],
                    marker='^', c='blue', s=80, label='Go-ICP Start Point')
        plt.scatter(estimated_points_go_icp[-1, 0], estimated_points_go_icp[-1, 1],
                    marker='^', c='red', s=80, label='Go-ICP End Point')

    plt.title("Map in Real-World Coordinates")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

# !! ADAPT TO YOUR PATH !!
image = iio.imread("NodeCodeImpressive/full_room_01062025.png")

px_resolution = 0.05  # m

t_mapOG_px = np.array([23, 138])            # px
t_mapOG = t_mapOG_px * px_resolution        # m

theta = 6.888258277       # deg
alpha = 180 + theta       # deg

T_map_to_mapOG = np.array([
    [np.cos(np.deg2rad(alpha)), -np.sin(np.deg2rad(alpha)), t_mapOG[0]],
    [np.sin(np.deg2rad(alpha)), np.cos(np.deg2rad(alpha)), t_mapOG[1]],
    [0, 0, 1]
])

def transform_point(point, T):
    """
    Transform a point using a transformation matrix.
    
    Args:
        point (np.array): A 2D point as a numpy array [x, y].
        T (np.array): A 3x3 transformation matrix.
    
    Returns:
        np.array: The transformed point as a numpy array [x', y'].
    """
    point_homogeneous = np.array([point[0], point[1], 1])  # Convert to homogeneous coordinates
    transformed_point = T @ point_homogeneous  # Matrix multiplication
    return transformed_point[:2]  # Return only the x and y coordinates


# Coordinate convention: blackboard means x axis, window means y axis

# --------- First path --------- 
rectangle_pt1 = np.array([5.21, 2.02])          # Bottom-left corner
rectangle_pt2 = np.array([0.71, 2.02])          # Bottom-right corner
rectangle_pt3 = np.array([0.71, 4.27])          # Top-right corner
rectangle_pt4 = np.array([5.21, 4.27])          # Top-left corner
rectangle_pt5 = np.array([5.21, 2.02])          # Bottom-left corner
rectangle_points = np.array([rectangle_pt1, rectangle_pt2, rectangle_pt3, rectangle_pt4, rectangle_pt5])

# --------- Second path --------- 
# 12 points in the complex shape
complex1_pt1 = np.array([2.51, 6.67])          
complex1_pt2 = np.array([4.31, 6.67])          
complex1_pt3 = np.array([5.21, 5.77])          
complex1_pt4 = np.array([5.21, 4.27])  

complex1_pt5 = np.array([4.01, 4.97])
complex1_pt6 = np.array([3.11, 3.97])
complex1_pt7 = np.array([1.91, 4.57])
complex1_pt8 = np.array([0.71, 4.27])

complex1_pt9 = np.array([0.71, 2.02])
complex1_pt10 = np.array([5.21, 2.02])
complex1_pt11 = np.array([5.21, 4.27])
complex1_pt12 = np.array([6.41, 2.47])

complex1_points = np.array([complex1_pt1, complex1_pt2, complex1_pt3, complex1_pt4,
                             complex1_pt5, complex1_pt6, complex1_pt7, complex1_pt8,
                             complex1_pt9, complex1_pt10, complex1_pt11, complex1_pt12])

# --------- Third path --------- 
# 19 points in the complex shape
complex2_pt1 = np.array([0.71, 4.27])
complex2_pt2 = np.array([0.71, 2.02])
complex2_pt3 = np.array([2.51, 2.02])          
complex2_pt4 = np.array([3.71, 1.12])  

complex2_pt5 = np.array([3.71, 1.87])
complex2_pt6 = np.array([2.51, 2.02])
complex2_pt7 = np.array([0.71, 2.02])
complex2_pt8 = np.array([0.71, 4.27])

complex2_pt9 = np.array([3.11, 3.97])
complex2_pt10 = np.array([4.01, 4.87])
complex2_pt11 = np.array([4.31, 4.27])
# complex2_pt12 = np.array([5.21, 4.27])

complex2_pt13 = np.array([4.61, 6.07])
complex2_pt14 = np.array([6.11, 6.07])
complex2_pt15 = np.array([4.31, 4.27])
complex2_pt16 = np.array([4.91, 3.07])
complex2_pt17 = np.array([5.21, 2.02])
complex2_pt18 = np.array([6.41, 2.47])
complex2_pt19 = np.array([5.21, 4.27])

complex2_points = np.array([complex2_pt1, complex2_pt2, complex2_pt3, complex2_pt4,
                             complex2_pt5, complex2_pt6, complex2_pt7, complex2_pt8,
                             complex2_pt9, complex2_pt10, complex2_pt11,
                             complex2_pt13, complex2_pt14, complex2_pt15, complex2_pt16,
                             complex2_pt17, complex2_pt18, complex2_pt19])

# Transform the rectangle points using the transformation matrix
# transformed_points = np.array([transform_point(pt, T_map_to_mapOG) for pt in rectangle_points])
# transformed_points = np.array([transform_point(pt, T_map_to_mapOG) for pt in complex1_points])
transformed_points = np.array([transform_point(pt, T_map_to_mapOG) for pt in complex2_points])

# --- Load estimated trajectory from CSV file ---
csv_path_pure = "Final_Data/0_ICP_Pure_ONLY/01_Bad_Init/bag3_downsample.csv"
csv_path_go_icp = "Final_Data/3_GO_ICP/01_Bad_Init/bag3_downsample_cut.csv"

df_pure = pd.read_csv(csv_path_pure)
df_go_icp = pd.read_csv(csv_path_go_icp)

n = 10  # Change this to sample every n-th point
estimated_points_pure = df_pure[['x', 'y']].to_numpy()[::n]
estimated_points_go_icp = df_go_icp[['x', 'y']].to_numpy()[::n]

# --- Plot and display the transformed points ---
print("Transformed Rectangle Points:")
for pt in transformed_points:
    print(pt)

display_map_with_points_map_frame(image, transformed_points, [-10, -10], px_resolution, estimated_points_pure, estimated_points_go_icp)

