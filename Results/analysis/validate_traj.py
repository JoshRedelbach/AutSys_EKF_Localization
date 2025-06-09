import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error

# Provided control points
control_points_path_1 = np.array([
    # [-3.78012847,  4.26972752],
    [0.68739027, 4.80942777],
    [0.95724039, 2.57566841],
    [-3.51027834,  2.03596816],
    [-3.78012847,  4.26972752]
])

control_points_path_2 = np.array([
    [-0.54192697, -0.02288835],
    [-2.32893446, -0.23876845],
    [-3.33037826,  0.54679524],
    [-3.51027834,  2.03596816],
    [-2.23498664,  1.48494086],
    [-1.46141628,  2.58566286],
    [-0.19811792,  2.13391376],
    [0.95724039, 2.57566841],
    [0.68739027, 4.80942777],
    [-3.78012847,  4.26972752],
    [-3.51027834,  2.03596816],
    [-4.91749677,  3.67905558]
])

control_points_path_3 = np.array([
    # [0.95724039, 2.57566841],
    [0.68739027, 4.80942777],
    [-1.09961723,  4.59354767],
    [-2.39889561,  5.34313135],
    [-2.30894556,  4.5985449 ],
    [-1.09961723,  4.59354767],
    [0.68739027, 4.80942777],
    [0.95724039, 2.57566841],
    [-1.46141628,  2.58566286],
    [-2.24697998,  1.58421906],
    [-2.6167746,   2.14390821],
    [-2.69872908,  0.3209207 ],
    [-4.18790199,  0.14102061],
    [-2.6167746,   2.14390821],
    [-3.35636383,  3.2632865 ],
    [-3.78012847,  4.26972752],
    [-4.91749677,  3.67905558],
    [-3.51027834,  2.03596816]
])

control_points_kidnapping = np.array([
    [-3.78012847,  4.26972752],
    [0.68739027, 4.80942777],
    [-0.54192697, -0.02288835],
    [-2.32893446, -0.23876845],
    [-3.33037826,  0.54679524],
    [-3.51027834,  2.03596816],
    [-2.23498664,  1.48494086],
    [-1.46141628,  2.58566286],
    [-0.19811792,  2.13391376],
    [0.95724039, 2.57566841]]
)

# Matching function with sequential constraint and fixed last point
def match_control_points_sequentially(control_points, estimated_positions):
    matched_estimated_points = []
    matched_indices = set()
    last_used_index = -1

    for i, cp in enumerate(control_points):
        if i == len(control_points) - 1:
            matched_estimated_points.append(estimated_positions[-1])
            matched_indices.add(len(estimated_positions) - 1)
            break

        dists = np.linalg.norm(estimated_positions - cp, axis=1)
        candidate_indices = [i for i in np.argsort(dists) if i > last_used_index and i not in matched_indices]

        if candidate_indices:
            best_idx = candidate_indices[0]
        else:
            fallback_idx = [i for i in np.argsort(dists) if i not in matched_indices][0]
            best_idx = fallback_idx

        matched_estimated_points.append(estimated_positions[best_idx])
        matched_indices.add(best_idx)
        last_used_index = best_idx

    return np.array(matched_estimated_points)

# Select desired control points based on the path
control_points = control_points_path_3  # Change to control_points_path_2 or control_points_path_3 as needed

# Load estimated trajectory from CSV
csv_path = "Final_Data/5_Cam/ekf_trajectory_bad_initial_Rcam_01_final.csv"
estimated_df = pd.read_csv(csv_path)

# Extract estimated positions (x, y)
estimated_positions = estimated_df[['x', 'y']].to_numpy()

# Apply sequential matching
matched_estimated_points_sequential = match_control_points_sequentially(control_points, estimated_positions)

# Compute ATE
ate_errors_sequential = np.linalg.norm(matched_estimated_points_sequential - control_points, axis=1)
mean_ate_seq = np.mean(ate_errors_sequential)
rmse_ate_seq = np.sqrt(mean_squared_error(control_points, matched_estimated_points_sequential))
max_ate_seq = np.max(ate_errors_sequential)

# Extract corresponding timestamps
matched_indices_seq = [np.where((estimated_positions == pt).all(axis=1))[0][0]
                       for pt in matched_estimated_points_sequential]
matched_timestamps = estimated_df.iloc[matched_indices_seq]['time'].values
is_strictly_increasing = np.all(np.diff(matched_timestamps) > 0)

# Create result dataframe
matched_coords_and_timestamps = pd.DataFrame({
    "time": matched_timestamps,
    "x": matched_estimated_points_sequential[:, 0],
    "y": matched_estimated_points_sequential[:, 1],
    "ATE [m]": ate_errors_sequential
})

print("Matched Coordinates and Timestamps:")
print(matched_coords_and_timestamps.round(3))

metrics_overview = pd.DataFrame({
    "Mean ATE [m]": [mean_ate_seq],
    "RMSE ATE [m]": [rmse_ate_seq],
    "Max ATE [m]": [max_ate_seq],
    "Timestamps Increasing": [is_strictly_increasing]
})

print("\nMetrics Overview:")
print(metrics_overview.round(3))
