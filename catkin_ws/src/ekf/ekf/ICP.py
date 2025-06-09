import numpy as np
from scipy.spatial import KDTree
import time

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps points A to B.
    Returns the transformation matrix T (3x3).
    Args:
        A: Nx2 array of points (source)
        B: Nx2 array of points (destination)
    Returns:
        T: 3x3 transformation matrix that maps A to B
    The transformation matrix T is in the form:
    | R t |
    | 0 1 |
    where R is a 2x2 rotation matrix and t is a translation vector.
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    T = np.identity(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def downsample_scan(scan_points, step=2):
    """
    Downsamples the scan points to a maximum of max_points.
    Args:
        scan_points: Nx2 array of points
        step: downsampling step size
    Returns:
        downsampled points: array of points with shape (N//step, 2)
    """
    return scan_points[::step]


def transform_points(points, T):
    """
    Applies a 3x3 homogeneous transformation matrix T to Nx2 points.
    Args:
        points: Nx2 array of points
        T: 3x3 transformation matrix
    Returns:
        transformed: Nx2 array of transformed points
    """
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    transformed = (T @ points_hom.T).T
    return transformed[:, :2]


def icp_pure(src, dst, max_iter=20, tol=1e-4, dst_tree=None):
    """
    Iterative Closest Point (ICP) algorithm to align source points to destination points.
    Args:
        src: Nx2 array of source points
        dst: Mx2 array of destination points
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        dst_tree: KDTree for destination points (optional; built internally if not provided)
    Returns:
        T_total: final 3x3 transformation matrix
        error: final mean alignment error
    """
    # Check if dst_tree is provided, if not, create it
    if dst_tree is None:
        dst_tree = KDTree(dst)
    
    # Initialize transformation matrix
    T_total = np.eye(3)

    # Initialize previous error for convergence check
    prev_error = float('inf')

    # Iterate until convergence or max iterations
    for _ in range(max_iter):
        # Find nearest neighbors in destination for each source point
        dists, indices = dst_tree.query(src, workers=-1)
        # Get matched destination points
        dst_matched = dst[indices]
        # Search for the best fit transformation between src and dst_matched
        T = best_fit_transform(src, dst_matched)
        # Apply the transformation to the source points
        src = transform_points(src, T)
        # Update the total transformation matrix
        T_total = T @ T_total
        # Calculate the mean alignment error
        error = np.mean(np.linalg.norm(dst_matched - src, axis=1))
        # Check for convergence
        if abs(prev_error - error) < tol:
            break
        # Update previous error
        prev_error = error

    return T_total, error


def icp_ransac_nested(src, dst, max_iter=10, dst_tree=None, ransac_iter=10, ransac_samples=4, ransac_threshold=0.3):
    """
    Runs ICP with RANSAC in every iteration to find a robust transformation between source and destination points.
    Args:
        src: Nx2 array of source points
        dst: Mx2 array of destination points
        max_iter: maximum number of ICP iterations
        dst_tree: KDTree of destination points (optional; built internally if not provided)
        ransac_iter: number of RANSAC iterations
        ransac_samples: number of point pairs to sample per RANSAC iteration
        ransac_threshold: distance threshold to count inliers
    Returns:
        T_total: final 3x3 transformation matrix
        prev_error: mean alignment error after ICP refinement
    """
    # Check if dst_tree is provided, if not, create it
    if dst_tree is None:
        dst_tree = KDTree(dst)
    
    # Initialize transformation matrix and previous error
    T_total = np.eye(3)
    prev_error = float('inf')

    # Iterate for the specified number of ICP steps
    for icp_step in range(max_iter):
        # Find nearest neighbors in destination for each source point
        dists, indices = dst_tree.query(src, workers=-1)
        # Get matched destination points
        dst_matched = dst[indices]

        # Initialize RANSAC variables
        best_T = None
        max_inliers = 0
        best_error = float('inf')

        # RANSAC loop
        for _ in range(ransac_iter):
            # Randomly sample points for RANSAC
            sample_indices = np.random.choice(len(src), ransac_samples, replace=False)
            A_sample = src[sample_indices]
            B_sample = dst_matched[sample_indices]

            # Compute the best fit transformation for the sampled points
            T = best_fit_transform(A_sample, B_sample)
            # Apply the transformation to the source points
            src_transformed = transform_points(src, T)

            # Calculate distances to the matched destination points
            # and determine inliers based on the RANSAC threshold
            inlier_dists = np.linalg.norm(dst_matched - src_transformed, axis=1)
            inliers = inlier_dists < ransac_threshold
            num_inliers = np.sum(inliers)

            # If the number of inliers is greater than the current maximum, update best transformation
            if num_inliers > max_inliers:
                refined_T = best_fit_transform(src[inliers], dst_matched[inliers])
                src_refined = transform_points(src, refined_T)
                error = np.mean(np.linalg.norm(dst_matched[inliers] - src_refined[inliers], axis=1))
                best_T = refined_T
                max_inliers = num_inliers
                best_error = error

        if best_T is None:
            print("Warning: No valid RANSAC transformation found in iteration", icp_step)
            break
        
        # Apply the best transformation found by RANSAC
        src = transform_points(src, best_T)
        T_total = best_T @ T_total

        # Check for convergence
        if abs(prev_error - best_error) < 1e-4:
            break

        prev_error = best_error

    return T_total, prev_error


def icp_ransac_first(src, dst, max_iter=10, dst_tree=None, ransac_iter=10, ransac_samples=4, ransac_threshold=0.3, percentage_inliers=0.6):
    """
    Runs RANSAC first to find a robust initial transformation, then refines it with ICP.
    Args:
        src: Nx2 array of source points
        dst: Mx2 array of destination points
        max_iter: maximum number of ICP iterations
        dst_tree: KDTree of destination points (optional; built internally if not provided)
        ransac_iter: number of RANSAC iterations
        ransac_samples: number of point pairs to sample per RANSAC iteration
        ransac_threshold: distance threshold to count inliers
        percentage_inliers: minimum percentage of inliers required to consider a transformation valid
    Returns:
        T_final: final 3x3 transformation matrix
        final_error: mean alignment error after ICP refinement
    """
    # Check if dst_tree is provided, if not, create it
    if dst_tree is None:
        dst_tree = KDTree(dst)

    # --- Step 1: RANSAC ---
    # Find nearest neighbors in destination for each source point
    dists, indices = dst_tree.query(src, workers=-1)
    
    # Get matched destination points
    dst_matched = dst[indices]

    # Initialize variables for RANSAC
    best_n_inlier = len(src) * percentage_inliers
    best_T_ransac = None
    best_inliers_mask = None
    max_inliers = 0

    # Initialize variables for RANSAC
    for _ in range(ransac_iter):
        # Randomly sample points for RANSAC
        sample_idx = np.random.choice(len(src), ransac_samples, replace=False)
        A_sample = src[sample_idx]
        B_sample = dst_matched[sample_idx]

        # Compute the best fit transformation for the sampled points
        T = best_fit_transform(A_sample, B_sample)
        
        # Apply the transformation to the source points
        src_transformed = transform_points(src, T)

        # Calculate distances to the matched destination points
        # and determine inliers based on the RANSAC threshold
        dists_all = np.linalg.norm(dst_matched - src_transformed, axis=1)
        inliers = dists_all < ransac_threshold
        num_inliers = np.sum(inliers)

        # If the number of inliers is greater than the current maximum, update best transformation
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_T_ransac = T
            best_inliers_mask = inliers

    # Check if we found a valid transformation with sufficient inliers
    if best_T_ransac is None or max_inliers < best_n_inlier:
        print("Warning: RANSAC failed to find a good alignment.")
        return np.eye(3), float('inf')

    # --- Step 2: Filter to inliers ---
    # Get the inliers from the best transformation found by RANSAC
    src_inliers = src[best_inliers_mask]
    src_inliers = transform_points(src_inliers, best_T_ransac)

    # --- Step 3: Refine using standard ICP on inliers ---
    refined_T, final_error = icp_pure(src_inliers, dst, max_iter=max_iter, dst_tree=dst_tree)

    # Combine RANSAC and ICP transformations
    T_final = refined_T @ best_T_ransac

    return T_final, final_error


def grid_search_go_icp(src_points, dst_points, rot_range=(-np.pi, np.pi), rot_step=np.deg2rad(90), max_icp_iter=10, dst_tree=None):
    """
    Performs a grid search for the best initial pose using GO-ICP.
    Args:
        src_points: Nx2 array of source points
        dst_points: Mx2 array of destination points
        rot_range: tuple (min_angle, max_angle) for rotation angles in radians
        rot_step: step size for rotation angles in radians
        max_icp_iter: maximum number of ICP iterations
        dst_tree: KDTree of destination points (optional; built internally if not provided)
    Returns:
        best_T: 3x3 transformation matrix that minimizes the error
        lowest_error: minimum mean alignment error found
        flag: status flag (for debugging)
    """
    # Check if dst_tree is provided, if not, create it
    if dst_tree is None:
        dst_tree = KDTree(dst_points)
    
    # Initialize variables
    best_T = None
    lowest_error = float('inf')
    flag = 1

    # Define rotation angles for grid search
    angles = np.arange(rot_range[0], rot_range[1], rot_step)
    
    # Define grid points for translation
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

    # Perform grid search over rotation and translation
    for theta in angles:
        # Create rotation matrix for the current angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        # Iterate over grid points for translation
        for pt in grid_points:
            # Create initial guess for the transformation matrix
            T_guess = np.eye(3)
            T_guess[:2, :2] = R
            T_guess[:2, 2] = [pt[0], pt[1]]
            # Apply the initial guess to the source points
            src_transformed = transform_points(src_points, T_guess)
            # Run ICP with the current guess
            T_icp, err = icp_pure(src_transformed, dst_points, max_iter=max_icp_iter, dst_tree=dst_tree)
            # Combine the ICP result with the initial guess
            T_final = T_icp @ T_guess
            # Check if the error is lower than the current lowest error
            # and update the best transformation if so
            if err < lowest_error:
                lowest_error = err
                best_T = T_final
            # Check if the error is below a threshold to consider it a valid transformation
            if lowest_error < 1e-4:
                return best_T, lowest_error, flag

    return best_T, lowest_error, flag


def robust_icp_pipeline(src_points, dst_points, init_pose=None, ransac_iter=10, ransac_samples=4, ransac_threshold=0.3, percentage_inliers=0.6, max_icp_iter=20, max_points=180):
    """
    Runs a robust ICP pipeline with option to call GO-ICP reinitialization.
    Args:
        src_points: Nx2 array of source points
        dst_points: Mx2 array of destination points
        init_pose: optional initial pose as a 3x3 transformation matrix
        ransac_iter: number of RANSAC iterations
        ransac_samples: number of point pairs to sample per RANSAC iteration
        ransac_threshold: distance threshold to count inliers
        percentage_inliers: minimum percentage of inliers required to consider a transformation valid
        max_icp_iter: maximum number of ICP iterations
        max_points: maximum number of points to keep after downsampling
    Returns:
        final_T: 3x3 transformation matrix that aligns src_points to dst_points
        error: mean alignment error after ICP refinement
        flag: status flag (for debugging)
    """
    # Create KDTree for destination points
    dst_tree = KDTree(dst_points)
    flag = 0

    # ---- Downsample Source Points ----
    if len(src_points) > max_points:
        # print("Max points: ", max_points)
        # print("Len source points before: ", len(src_points))
        src_points = downsample_scan(src_points, 2)
        # print("Len source points after: ", len(src_points))

    best_n_inlier = percentage_inliers * len(src_points)

    # Define a function to count inliers based on the RANSAC threshold
    def count_inliers(transformed):
        dists, _ = dst_tree.query(transformed, workers=-1)
        return np.sum(dists < ransac_threshold), dists

    # Apply Initial Pose if provided and count inliers
    if init_pose is not None:
        src_init = transform_points(src_points, init_pose)
        inliers_init, _ = count_inliers(src_init)
        T_best = init_pose
        best_src = src_init
    else:
        inliers_init = 0
        T_best = np.eye(3)
        best_src = src_points.copy()

    # Check if initial inliers are sufficient
    # If not --> start grid search for GO-ICP
    if inliers_init < best_n_inlier:
        # print(f"Only {inliers_init} inliers ---> Starting grid search for GO-ICP")
        # time_goicp_start = time.time()

        best_src, lowest_error, flag = grid_search_go_icp(src_points, dst_points, max_icp_iter=max_icp_iter, dst_tree=dst_tree)
        
        # time_goicp_end = time.time()
        # print(f"GO-ICP took {time_goicp_end - time_goicp_start:.7f} seconds")

        return best_src, lowest_error, flag
    # else:
        # # Debugging
        # print("Initial pose used with inliers: ", inliers_init)

    # Run pure ICP if initial inliers are sufficient
    T_refined, error = icp_pure(best_src, dst_points, max_iter=max_icp_iter, dst_tree=dst_tree)

    # Combine the initial pose with the refined transformation
    final_T = T_refined @ T_best

    return final_T, error, flag