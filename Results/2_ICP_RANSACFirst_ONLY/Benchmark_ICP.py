import numpy as np
from scipy.spatial import KDTree

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps points A to B.
    Returns the transformation matrix T (3x3).
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


def downsample_scan(scan_points,step=2): #,max_points=360
    """Downsamples the scan points to a maximum of max_points."""
    # step = max(1, len(scan_points) // max_points)
    return scan_points[::step]


def transform_points(points, T):
    """Applies a 3x3 homogeneous transformation matrix T to Nx2 points."""
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    transformed = (T @ points_hom.T).T
    return transformed[:, :2]


def icp_pure(src, dst, max_iter=20, tol=1e-4, dst_tree=None):
    T_total = np.eye(3)
    prev_error = float('inf')
    for _ in range(max_iter):
        dists, indices = dst_tree.query(src, workers=-1)
        dst_matched = dst[indices]
        T = best_fit_transform(src, dst_matched)
        src = transform_points(src, T)
        T_total = T @ T_total
        error = np.mean(np.linalg.norm(dst_matched - src, axis=1))
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    return T_total, error


def icp_ransac_nested(src, dst, max_iter=10, dst_tree=None, ransac_iter=10, ransac_samples=4, ransac_threshold=0.3):
    """
    ICP with RANSAC in each iteration.
    Parameters:
        src: Nx2 array of source points
        dst: Mx2 array of destination points
        max_iter: number of ICP iterations
        dst_tree: KDTree for destination points
        ransac_iter: RANSAC iterations per ICP step
        ransac_samples: number of point pairs to sample in each RANSAC
        ransac_threshold: inlier distance threshold
        best_n_inlier: minimum number of inliers to accept a RANSAC hypothesis
    Returns:
        T_total: final 3x3 transformation matrix
        error: final mean alignment error
    """
    T_total = np.eye(3)
    prev_error = float('inf')

    for icp_step in range(max_iter):
        dists, indices = dst_tree.query(src, workers=-1)
        dst_matched = dst[indices]

        best_T = None
        max_inliers = 0
        best_error = float('inf')

        for _ in range(ransac_iter):

            sample_indices = np.random.choice(len(src), ransac_samples, replace=False)
            A_sample = src[sample_indices]
            B_sample = dst_matched[sample_indices]

            T = best_fit_transform(A_sample, B_sample)
            src_transformed = transform_points(src, T)

            inlier_dists = np.linalg.norm(dst_matched - src_transformed, axis=1)
            inliers = inlier_dists < ransac_threshold
            num_inliers = np.sum(inliers)

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

        src = transform_points(src, best_T)
        T_total = best_T @ T_total

        if abs(prev_error - best_error) < 1e-4:
            break

        prev_error = best_error

    return T_total, prev_error


def icp_ransac_first(src, dst, max_iter=10, dst_tree=None, ransac_iter=10, ransac_samples=4, ransac_threshold=0.3, percentage_inliers=200):
    """
    Runs RANSAC once to eliminate outliers, then refines the transformation using ICP on inliers.
    
    Parameters:
        src: Nx2 array of source points
        dst: Mx2 array of destination points
        max_iter: maximum number of ICP iterations
        dst_tree: KDTree of dst (optional; built internally if not provided)
        ransac_iter: number of RANSAC iterations
        ransac_samples: number of point pairs to sample per RANSAC iteration
        ransac_threshold: distance threshold to count inliers
        best_n_inlier: minimum inliers required to accept the RANSAC result

    Returns:
        T_final: 3x3 transformation matrix
        final_error: mean alignment error after ICP refinement
    """

    # Step 1: Initial correspondences using nearest neighbors
    dists, indices = dst_tree.query(src, workers=-1)
    dst_matched = dst[indices]
    best_n_inlier = len(src)*percentage_inliers
    best_T_ransac = None
    best_inliers_mask = None
    max_inliers = 0

    for _ in range(ransac_iter):

        sample_idx = np.random.choice(len(src), ransac_samples, replace=False)
        A_sample = src[sample_idx]
        B_sample = dst_matched[sample_idx]

        T = best_fit_transform(A_sample, B_sample)
        src_transformed = transform_points(src, T)
        dists_all = np.linalg.norm(dst_matched - src_transformed, axis=1)
        inliers = dists_all < ransac_threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_T_ransac = T
            best_inliers_mask = inliers

    if best_T_ransac is None or max_inliers < best_n_inlier:
        print("Warning: RANSAC failed to find a good alignment.")
        return np.eye(3), float('inf')

    # Step 2: Filter to inliers
    src_inliers = src[best_inliers_mask]

    src_inliers = transform_points(src_inliers, best_T_ransac)

    # Step 3: Refine using standard ICP on inliers
    refined_T, final_error = icp_pure(src_inliers, dst, max_iter=max_iter, dst_tree=dst_tree)

    # Combine RANSAC and ICP transformations
    T_final = refined_T @ best_T_ransac

    return T_final, final_error


def grid_search_go_icp(src_points, dst_points, 
                       rot_range=(-np.pi, np.pi), rot_step=np.deg2rad(90),
                        max_icp_iter=10, dst_tree=None):
    best_T = None
    lowest_error = float('inf')
    flag = 1
    angles = np.arange(rot_range[0], rot_range[1], rot_step)
    
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

    grid_points = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8,pt9, pt10, pt11,pt12]) #pt12,pt13

    for theta in angles:
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        for pt in grid_points:
            T_guess = np.eye(3)
            T_guess[:2, :2] = R
            T_guess[:2, 2] = [pt[0], pt[1]]
            src_transformed = transform_points(src_points, T_guess)
            T_icp, err = icp_pure(src_transformed, dst_points, max_iter=max_icp_iter, dst_tree=dst_tree)
            T_final = T_icp @ T_guess
            if err < lowest_error:
                lowest_error = err
                best_T = T_final
            if lowest_error < 1e-4:
                return best_T, lowest_error

    return best_T, lowest_error, flag


def icp_pipeline(src_points, dst_points, init_pose=None, ransac_iter=10, ransac_samples=4, ransac_threshold=0.3, percentage_inliers=0.6, max_icp_iter=20, max_points=180):
    
    dst_tree = KDTree(dst_points)
    flag = 0

    # ---- Downsample Source Points ----
    if len(src_points) > max_points:
        src_points = downsample_scan(src_points, 2)

    # def count_inliers(transformed):
    #     dists, _ = dst_tree.query(transformed, workers=-1)
    #     return np.sum(dists < ransac_threshold), dists

    if init_pose is not None:
        src_init = transform_points(src_points, init_pose)
        #inliers_init, _ = count_inliers(src_init)
        T_best = init_pose
        best_src = src_init
    else:
        #inliers_init = 0
        T_best = np.eye(3)
        best_src = src_points.copy()

    # if inliers_init < best_n_inlier:
    #     print(f"Only {inliers_init} inliers ---> Starting grid search for GO-ICP")
    #     return grid_search_go_icp(src_points, dst_points, max_icp_iter=10, dst_tree=dst_tree)
    # else:
    #     print("Initial pose used with inliers: ", inliers_init)

    # T_refined, error = icp_pure(best_src, dst_points, max_iter=max_icp_iter, dst_tree=dst_tree)

    # T_refined, error = icp_ransac_nested(best_src, dst_points, max_iter=10, dst_tree=dst_tree, ransac_iter=5, ransac_samples=ransac_samples, ransac_threshold=ransac_threshold)

    T_refined, error = icp_ransac_first(best_src, dst_points, max_iter=20, dst_tree=dst_tree, ransac_iter=5, ransac_samples=ransac_samples, ransac_threshold=ransac_threshold, percentage_inliers=percentage_inliers)

    final_T = T_refined @ T_best

    return final_T, error, flag