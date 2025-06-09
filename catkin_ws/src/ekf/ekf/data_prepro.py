#!/usr/bin/env python3
import numpy as np
from tf.transformations import euler_from_quaternion
from PIL import Image
import yaml


def scan_to_points(scan_msg):
    """
    Convert a ROS LaserScan message to a 2D numpy array of points.
    Args:
        scan_msg: A ROS LaserScan message.
    Returns:
        A numpy array of shape (N, 2) where N is the number of valid points.
        Each row contains [x, y] coordinates in the robot's frame.
    """
    # Get angles and ranges from the scan message
    angles = scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
    ranges = np.array(scan_msg.ranges)

    # Filter out invalid ranges (e.g., zero or NaN)
    valid = (ranges > 0.1) & np.isfinite(ranges)
    ranges = ranges[valid]
    angles = angles[valid]

    # Convert polar to Cartesian
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    points = np.vstack((x, y)).T  # shape (N, 2)
    return points


def odom_msg_to_pose(msg):
    """
    Convert a ROS Odometry message to a numpy array of the robot's pose.
    Args:
        msg: A ROS Odometry message.
    Returns:
        A numpy array of shape (3,) containing [x, y, theta] in the robot's frame.
    """
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    q = msg.pose.pose.orientation
    quaternion = [q.x, q.y, q.z, q.w]
    _, _, theta = euler_from_quaternion(quaternion)
    
    return np.array([x, y, theta])


def load_map(pgm_file, yaml_file):
    """
    Load a map from a PGM image and its corresponding YAML metadata file.
    Args:
        pgm_file: Path to the PGM image file.
        yaml_file: Path to the YAML metadata file.
    Returns:
        A numpy array of shape (N, 2) containing the coordinates of occupied pixels in the map.
    """
    with open(yaml_file, 'r') as f:
        map_metadata = yaml.safe_load(f) # Load map metadata

    resolution = map_metadata['resolution']  # meters per pixel
    origin = map_metadata['origin']          # [x, y, theta]
    negate = map_metadata.get('negate', 0)   # 0 or 1
    occupied_thresh = map_metadata.get('occupied_thresh', 0.65)

    # Load pgm image
    img = Image.open(pgm_file)
    map_data = np.array(img)

    # Convert pixels to [0,1] normalized values (assuming 0=occupied, 255=free)
    if negate:
        normalized = map_data / 255.0
    else:
        normalized = 1 - (map_data / 255.0)

    # Occupied pixels mask
    occupied = normalized > occupied_thresh

    # Extract coordinates of occupied pixels
    ys, xs = np.where(occupied)  # pixel indices where map is occupied
    height = map_data.shape[0]

    # Convert pixel indices to meters (map frame)
    # origin[0], origin[1] is bottom-left corner of map in world frame
    x_world = xs * resolution + origin[0]
    y_world = (height - ys) * resolution + origin[1]

    points = np.vstack([x_world, y_world]).T
    return points