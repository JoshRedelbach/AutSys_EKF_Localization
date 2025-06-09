#!/usr/bin/env python3

# --- Import necessary libraries ---  
import numpy as np
import threading
import time
import cv2

# -- Import custom modules ---
import ekf.ICP as ICP
import ekf.Benchmark_ICP as BICP
import ekf.cam as cam

# --------------- CONSTANTS ---------------
# Parameeters for RANSAC and ICP
RANSAC_MAX_ITERATIONS = 10
RANSAC_THRESHOLD = 0.5
RANSAC_ITER = 10
RANSAC_SAMPLES = 4

# --------------- FUNCTIONS ---------------
# Function to normalize angles to the range [-pi, pi]
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def move_odom(x, u):
    """
    Update the state based on the odometry input.
    Args:
        x (np.array): Current state [x_pos, y_pos, theta].
        u (tuple): Control input (x_bar_before, x_bar_after) where each is a tuple (x, y, theta).
    Returns:
        np.array: Updated state [x_pos, y_pos, theta].
        float: delta_rot1 (rotation before translation).
        float: delta_trans (translation).
        float: delta_rot2 (rotation after translation).
    """
    # Unpack the control input and current state
    x_bar_before, x_bar_after = u
    x_pos, y_pos, theta = x

    # Calculate the change in position and orientation
    delta_rot1 = normalize_angle(
        np.arctan2(x_bar_after[1] - x_bar_before[1], x_bar_after[0] - x_bar_before[0]) - x_bar_before[2]
    )
    delta_trans = np.sqrt((x_bar_after[0] - x_bar_before[0])**2 + (x_bar_after[1] - x_bar_before[1])**2)
    delta_rot2 = normalize_angle(x_bar_after[2] - x_bar_before[2] - delta_rot1)

    # Update the state
    x_pos += delta_trans * np.cos(theta + delta_rot1)
    y_pos += delta_trans * np.sin(theta + delta_rot1)
    theta += delta_rot1 + delta_rot2
    theta = normalize_angle(theta)

    return np.array([x_pos, y_pos, theta]), delta_rot1, delta_trans, delta_rot2



# --------------- EKF CLASS ---------------
class EKF:
    def __init__(self, mu_init, Sigma_init, map_points, Q=None, R_icp=None, R_cam=None, tolerance_icp=1.0, tolerance_cam=1.0):
        """
        Initialize the Extended Kalman Filter (EKF) with initial state, covariance, and map points.
        Args:
            mu_init (np.array): Initial state estimate (x, y, theta).
            Sigma_init (np.array): Initial covariance matrix.
            map_points (np.array): Map points for the EKF update.
            Q (np.array): Process noise covariance matrix.
            R_icp (np.array): Measurement noise covariance for ICP updates.
            R_cam (np.array): Measurement noise covariance for camera updates.
            tolerance_icp (float): Tolerance for ICP updates.
            tolerance_cam (float): Tolerance for camera updates.
        """
        self.lock = threading.Lock()  # Protect shared state

        # Initialize state and covariance
        self.mu = mu_init
        self.Sigma = Sigma_init

        # Map points for ICP updates
        self.map_points = map_points

        # Initialize the tolerance for matching in udpate steps
        self.tolerance_icp = tolerance_icp
        self.tolerance_cam = tolerance_cam

        # Initialize noise covariance matrices
        self.Q = Q if Q is not None else np.diag([0.05, 0.05, np.deg2rad(2)])**2
        self.R_icp = R_icp if R_icp is not None else np.diag([0.1, 0.1, np.deg2rad(2)])**2
        self.R_cam = R_cam if R_cam is not None else np.diag([1, 1, np.deg2rad(2)])**2

        # Helper variables for updates and to collect statistics
        self.rejection = 0  # Counter for consecutive rejections of ICP

        self.number_of_updates_icp = 0
        self.runtime_total_updates_icp = 0
        self.number_of_updates_cam = 0
        self.runtime_total_updates_cam = 0

        self.number_of_rejections_icp = 0
        self.number_of_rejections_cam = 0

        # Camera related attributes
        self.last_keypoints = None
        self.last_descriptors = None
        self.time_last_updated = None
        self.last_v = None


    def predict(self, u, velocity_norm):
        """
        Predict the next state based on the odometry input.
        Args:
            u (tuple): Control input (x_bar_before, x_bar_after) where each is a tuple (x, y, theta).
            velocity_norm (float): Norm of the velocity vector for the robot.
        Returns:
            np.array: Updated state estimate (x, y, theta).
            np.array: Updated covariance matrix.
        """
        with self.lock:
            # Call the move_odom function to update the state based on odometry
            mu_bar, delta_rot1, delta_trans, _ = move_odom(self.mu, u)

            # Update the covariance matrix based on the motion model
            F = np.array([
                [1, 0, -delta_trans * np.sin(self.mu[2] + delta_rot1)],
                [0, 1,  delta_trans * np.cos(self.mu[2] + delta_rot1)],
                [0, 0, 1]
            ])
            Sigma_bar = F @ self.Sigma @ F.T + self.Q

            self.mu = mu_bar
            self.Sigma = Sigma_bar

            # Update the last velocity for the camera update
            self.last_v = velocity_norm

            return self.mu, self.Sigma


    def update_ICP(self, scan_points):
        """
        Update the state with a measurement using ICP on the laser scan points.
        Args:
            scan_points (np.array): Points from the laser scan in the robot's frame.
        Returns:
            np.array: Updated state estimate (x, y, theta).
            np.array: Updated covariance matrix.
        """
        with self.lock:
            # Check if the map points are available and downsample if necessary
            if len(self.map_points) > 5000:
                sampled_map = self.map_points[np.random.choice(len(self.map_points), 5000, replace=False)]
            else:
                sampled_map = self.map_points

            # Create initial guess for the ICP transformation based on the current state
            init_T_guess = np.array([
                [np.cos(self.mu[2]), -np.sin(self.mu[2]), self.mu[0]],
                [np.sin(self.mu[2]),  np.cos(self.mu[2]), self.mu[1]],
                [0,              0,             1]
            ])
            
        
            time_start_icp = time.time()

            # -- !! SELECT THE ICP YOU WANT !! --
            # 1 - Use the robust ICP Pipeline
            T_icp, error, flag = ICP.robust_icp_pipeline(scan_points, sampled_map, init_pose=init_T_guess, ransac_iter=RANSAC_ITER, ransac_samples=RANSAC_SAMPLES, ransac_threshold=RANSAC_THRESHOLD, percentage_inliers=0.6, max_icp_iter=20, max_points=180)            
            # 2 - Use the ICP Only Pipeline
            # T_icp, error, flag = BICP.icp_pipeline(scan_points, sampled_map, init_pose=init_T_guess, ransac_iter=RANSAC_ITER, ransac_samples=RANSAC_SAMPLES, ransac_threshold=RANSAC_THRESHOLD, percentage_inliers=0.6, max_icp_iter=20, max_points=180)
            
            # Collect statistics for the ICP update
            time_end_icp = time.time()
            delta_time = time_end_icp - time_start_icp
            self.runtime_total_updates_icp += delta_time 
            self.number_of_updates_icp += 1

            # T_icp is the transformation matrix from the laser to the map frame
            # -> we need the transformation from robot frame to map frame
            # -> include the transformation from robot frame to laser frame
            T_robot_to_laser = np.array([
                [1, 0, 0.07],
                [0, 1,    0],
                [0, 0,    0]
            ])
            T_icp = T_robot_to_laser @ T_icp

            # Extract the estimated pose from the transformation matrix
            R_est = T_icp[:2, :2]
            t_est = T_icp[:2, 2]
            angle_est = np.arctan2(R_est[1, 0], R_est[0, 0])
            z_obs = np.array([t_est[0], t_est[1], angle_est])

            y = z_obs - self.mu
            y[2] = normalize_angle(y[2])

            # Determine Mahalanobis distance and update the state
            H = np.eye(3)
            S = H @ self.Sigma @ H.T + self.R_icp
            d2 = y.T @ np.linalg.inv(S) @ y  # Mahalanobis distance
            # print(f"Mahalanobis distance: {d2:.3f}")

            if d2 <= self.tolerance_icp:
                K = self.Sigma @ H.T @ np.linalg.inv(S)
                self.mu = self.mu + K @ y
                self.mu[2] = normalize_angle(self.mu[2])
                self.Sigma = (np.eye(3) - K @ H) @ self.Sigma
                # print(f"Mahalanobis distance: {d2:.3f}")
                self.rejection = 0
            else:
                self.rejection = self.rejection+1
                # print("Update rejected due to large innovation. Rejection count: ", self.rejection)
            
            # Check if the number of rejections exceeds a threshold meaning the filter needs to be reset
            if self.rejection>5 or flag==1:
                print("Reset sigma. Old sigma: ", self.Sigma)
                self.Sigma = np.eye(3) * 5 # reset sigma
                self.rejection=0
                self.number_of_rejections_icp += 1

            return self.mu, self.Sigma


    def update_CAM(self, image, time_now):
        """
        Update the state with a measurement using camera pose estimation.
        Args:
            image (np.array): Image from the camera.
            time_now (float): Current time for timestamping the update.
        Returns:
            np.array: Updated state estimate (x, y, theta).
            np.array: Updated covariance matrix.
        """
        with self.lock:
            # If last velocity is None, it means that camera update is before odometry update
            # so udpate cannot be performed
            if self.last_v is None:
                return self.mu, self.Sigma
            
            # If last updated time is None, it means that this is the first camera update
            # so we set the delta time to None
            if self.time_last_updated is None:
                delta_time = None
            else:
                delta_time = time_now - self.time_last_updated

            time_start_cam = time.time()

            # Perform camera pose estimation
            z_obs, keypoints_current, descriptors_current = cam.image_pose_estimation(image, self.last_keypoints, self.last_descriptors, self.mu, self.last_v, delta_time)

            time_end_cam = time.time()
            delta_time = time_end_cam - time_start_cam
            self.runtime_total_updates_cam += delta_time 
            self.number_of_updates_cam += 1

            # Update the last updated time, keypoints, and descriptors
            self.time_last_updated = time_now
            self.last_keypoints = keypoints_current
            self.last_descriptors = descriptors_current
            
            if z_obs is None:
                return self.mu, self.Sigma      # No pose estimation done, return current state

            # Compute the innovation
            y = z_obs - self.mu
            y[2] = normalize_angle(y[2])

            # Determine Mahalanobis distance and update the state
            H = np.eye(3)
            S = H @ self.Sigma @ H.T + self.R_cam
            d2 = y.T @ np.linalg.inv(S) @ y  # Mahalanobis distance

            # print(f"Cam Update - Mahalanobis distance : {d2:.3f}")

            if d2 <= self.tolerance_cam:
                K = self.Sigma @ H.T @ np.linalg.inv(S)
                self.mu = self.mu + K @ y
                self.mu[2] = normalize_angle(self.mu[2])
                self.Sigma = (np.eye(3) - K @ H) @ self.Sigma            
            else:
                self.number_of_rejections_cam += 1

            return self.mu, self.Sigma


    def get_state(self):
        with self.lock:
            return self.mu.copy(), self.Sigma.copy()
