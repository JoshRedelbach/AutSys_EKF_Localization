import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import pandas as pd

def display_control_points_in_map(image, control_points, origin, resolution):
    """
    Displays the map image in the map (meter) coordinate frame,
    and plots given map points without converting them.

    Parameters:
    - image: 2D numpy array (map image, grayscale)
    - map_points: Nx2 numpy array of (x, y) points in meters (map frame)
    - origin: (x0, y0): bottom-left of map image in meters
    - resolution: meters per pixel
    - estimated_points: optional Nx2 numpy array of (x, y) points (estimated trajectory)

    """
    x0, y0 = origin
    height, width = image.shape[:2]

    # Compute image bounds in map frame (meters)
    x1 = x0 + width * resolution
    y1 = y0 + height * resolution

    # Plot map image with real-world extent
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', extent=[x0, x1, y0, y1], origin='upper')
    
    plt.scatter(control_points[:, 0], control_points[:, 1], c='blue', s=30, label='Control Points for Reinitialization')

    # Final display
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

pt1 = np.array([1, 2])
pt2 = np.array([-1, 2])
pt3 = np.array([-3, 2])
pt4 = np.array([-4.5, 2])
pt5 = np.array([-3.5, 0.5])
pt6 = np.array([-2.0, 0.0])
pt7 = np.array([-0.5, 0.0])
pt8 = np.array([-4.0, 3.5])
pt9 = np.array([-4.0, 4.5])
pt10 = np.array([-1.5, 5.0])
pt11 = np.array([0.75, 4.5])
control_points = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11])

pt1 = np.array([1, 2])
pt2 = np.array([-1, 2])
pt3 = np.array([-3, 2])
pt4 = np.array([-4.5, 2])
pt5 = np.array([-3.5, 0.5])
pt6 = np.array([-2.0, 0.0])
pt7 = np.array([-0.5, 0.0])
pt8 = np.array([-4.0, 3.5])
pt9 = np.array([-3.0, 5.0])
pt10 = np.array([-1.5, 5.0])
pt11 = np.array([0.75, 4.5])
pt12 = np.array([-4.5, 0.0])
pt13 = np.array([-4.5,4.5])

grid_points = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8,pt9, pt10, pt11, pt12, pt13])

display_control_points_in_map(image, grid_points , [-10, -10], px_resolution)

