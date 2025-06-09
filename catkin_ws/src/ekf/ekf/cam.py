# --- Import necessary libraries --- 
import cv2
import numpy as np
from tf.transformations import euler_from_matrix


# --------------- HELPER FUNCTIONS ---------------
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --------------- CONSTANTS ---------------
# Camera intrinsic parameters
K = np.array([[504.9629545390832, 0.0, 320.0448609274721],
              [0.0, 502.2914546831238, 240.8836661448442],
              [0.0, 0.0, 1.0]], dtype=np.float32)
# Distortion coefficients
D = np.array([0.1601848742442538, -0.2853676227443593, 0.002275430232885984, -0.006638550546738463, 0.0], dtype=np.float32)
# Projection matrix
P = np.array([[514.5068969726562, 0.0, 315.9696842856902, 0.0],
              [0.0, 515.4712524414062, 241.7468675217115, 0.0],
              [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

# Transformation matrix from camera to robot frame
T_cam_ideal_to_robot = np.array([
    [0,  0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0,  0, 0, 1]
])
alpha = np.deg2rad(5.5)
Rot_mat = np.array([
    [1, 0, 0],
    [0, np.cos(alpha), -np.sin(alpha)],
    [0, np.sin(alpha), np.cos(alpha)]
], dtype=np.float32)
T_cam_real_to_cam_ideal = np.array([
    [Rot_mat[0, 0], Rot_mat[0, 1], Rot_mat[0, 2], 0],
    [Rot_mat[1, 0], Rot_mat[1, 1], Rot_mat[1, 2], 0],
    [Rot_mat[2, 0], Rot_mat[2, 1], Rot_mat[2, 2], 0],
    [0, 0, 0, 1]])
# Get transformation matrix from real camera frame to robot frame
T_cam_to_robot = T_cam_ideal_to_robot @ T_cam_real_to_cam_ideal


# --------------- FUNCTION ---------------
def image_pose_estimation(image, kp_last, des_last, last_pose, velocity, delta_time):
    """ 
    Given an image and keypoints with descriptors of previous image, estimate the relative pose of the camera. 
    Args:
        - image (np.ndarray): The current image frame in RGB format.
        - kp_last (list): List of keypoints from the previous image.
        - des_last (np.ndarray): Descriptors corresponding to the previous keypoints.
        - last_pose (np.ndarray): Last estimated pose of the robot in the form of a state vector [x, y, theta].
        - velocity (np.ndarray): Current velocity norm of the robot in robot frame.
        - delta_time (float): Time difference since the last feature detection in seconds.
    Returns:
        - np.ndarray: Estimated absolute pose of the camera in the form of a state vector [x, y, theta].
        - keypoints of the current image.
        - descriptors of the current image.
    """

    # print("\nEstimating pose from image...")
    
    # ---- 1. Detect and compute ORB features ---- 
    orb = cv2.ORB_create(50)
    kp_current, des_current = orb.detectAndCompute(image, None)

    if kp_current is None or des_current is None or velocity is None or last_pose is None or delta_time is None:
        return None, kp_current, des_current

    # ---- 2. Match descriptors with the last descriptors using FLANN or BFMatcher ---- 
    # Match descriptors using KNN and Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des_last, des_current, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    ratio_thresh = 0.75
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 5:
        return None, kp_current, des_current
    
    # Take only 20 best matches for efficiency
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:20]
    
    # print(f"Number of good matches: {len(good_matches)}")

    # Extract matched keypoints
    pts_last = np.float32([kp_last[m.queryIdx].pt for m in good_matches])
    pts_current = np.float32([kp_current[m.trainIdx].pt for m in good_matches])

    # Undistort points to remove lens distortion
    pts_last_ud = cv2.undistortPoints(np.expand_dims(pts_last, axis=1), K, D, P=K)
    pts_current_ud = cv2.undistortPoints(np.expand_dims(pts_current, axis=1), K, D, P=K)

    # ---- 4. Compute essential matrix and recover pose based on the matches ---- 
    E, mask = cv2.findEssentialMat(pts_last_ud, pts_current_ud, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # ---- 5. Recover relative pose from the essential matrix and self.last_v ---- 
    _, R, t, mask_pose = cv2.recoverPose(E, pts_last_ud, pts_current_ud, K)

    # ---- 6. Estimate absoltue pose from the relative pose and self.mu ---- 
    # 6.1 Transform the relative pose between two camera frames to relative pose between the two robot frames
    T_cam_rel = np.eye(4)
    T_cam_rel[:3, :3] = R
    T_cam_rel[:3, 3] = t.flatten()

    T_robot_rel = T_cam_to_robot @ T_cam_rel @ np.linalg.inv(T_cam_to_robot)

    R_robot = T_robot_rel[:3, :3]
    t_robot = T_robot_rel[:3, 3]

    delta_alpha, delta_beta, delta_theta = euler_from_matrix(R_robot)
    
    delta_theta = normalize_angle(delta_theta)
    delta_translation = t_robot[:2]

    if abs(delta_theta) >= np.deg2rad(30):
        return None, kp_current, des_current

    # 6.2 Add relative pose to the current state (self.mu) to get the absolute pose
    scale = velocity * delta_time
    if np.linalg.norm(delta_translation) == 0:
        return None, kp_current, des_current

    delta_xy_scaled = (delta_translation / np.linalg.norm(delta_translation)) * scale

    x_k, y_k, theta_k = last_pose
    dx, dy = delta_xy_scaled

    x_new = x_k + dx
    y_new = y_k + dy
    theta_new = theta_k + delta_theta

    # print(f"\nImage-based pose: x={x_new:.4f}, y={y_new:.4f}, theta={np.rad2deg(theta_new):.4f} deg")

    new_pose = np.array([x_new, y_new, theta_new])

    return new_pose, kp_current, des_current

