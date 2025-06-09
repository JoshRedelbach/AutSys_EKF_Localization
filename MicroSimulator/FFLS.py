#!/usr/bin/env python3
import numpy as np
from collections import defaultdict

def ray_segment_intersection(ray_origin, ray_direction, seg_a, seg_b):
    """
    Returns the distance from the ray origin to the intersection point with the segment,
    or None if there is no valid intersection.
    """
    x1, y1 = ray_origin
    dx1, dy1 = ray_direction
    x2, y2 = seg_a
    x3, y3 = seg_b

    dx2 = x3 - x2
    dy2 = y3 - y2

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-8:
        return None  # Parallel lines

    dx = x2 - x1
    dy = y2 - y1

    t = (dx * dy2 - dy * dx2) / denom  # Distance along ray
    s = (dx * dy1 - dy * dx1) / denom  # Segment param

    if 0 <= s <= 1 and t >= 0:
        return t
    return None


def build_wall_grid(walls, cell_size=0.5):
    """
    Builds a spatial hash grid for efficient wall lookup.
    Ensures each wall is represented in all cells it touches.
    """
    grid = defaultdict(list)

    def get_cell(x, y):
        return int(x // cell_size), int(y // cell_size)

    for wall in walls:
        (x0, y0), (x1, y1) = wall

        dx = x1 - x0
        dy = y1 - y0
        length = int(max(abs(dx), abs(dy)) / (cell_size / 10)) + 1
        xs = np.linspace(x0, x1, length)
        ys = np.linspace(y0, y1, length)

        cells_covered = set()
        for x, y in zip(xs, ys):
            cell = get_cell(x, y)
            if cell not in cells_covered:
                grid[cell].append(wall)
                cells_covered.add(cell)

    return grid


def dda_ray_cells(x0, y0, angle, grid_bounds, cell_size=0.5, max_range=3.5):
    """
    Computes the list of grid cells traversed by a ray using DDA (fine stepping).
    """
    cells = []
    dx = np.cos(angle)
    dy = np.sin(angle)
    t = 0
    while t < max_range:
        x = x0 + t * dx
        y = y0 + t * dy
        if not (0 <= x < grid_bounds[0] and 0 <= y < grid_bounds[1]):
            break
        cell = (int(x // cell_size), int(y // cell_size))
        if cell not in cells:
            cells.append(cell)
        t += cell_size / 5  # fine steps for accuracy
    return cells


def fast_lidar_scan(pose, grid, grid_bounds, cell_size=0.5,
                    angle_min=0, angle_max=2*np.pi, angle_increment=0.0175,
                    range_min=0.05, range_max=3.5):
    """
    Simulates a 2D lidar scan using accurate raycasting and spatial hashing.
    Returns closest range measurements for each angle.
    """
    x, y, theta = pose
    angles = np.arange(angle_min, angle_max, angle_increment)
    ranges = []

    for a in angles:
        ray_angle = theta + a
        dx = np.cos(ray_angle)
        dy = np.sin(ray_angle)
        ray_dir = (dx, dy)

        ray_cells = dda_ray_cells(x, y, ray_angle, grid_bounds, cell_size, max_range=range_max)

        min_range = None
        for cell in ray_cells:
            if cell not in grid:
                continue
            for wall in grid[cell]:
                r = ray_segment_intersection((x, y), ray_dir, wall[0], wall[1])
                if r is not None and range_min <= r <= range_max:
                    if min_range is None or r < min_range:
                        min_range = r

        if min_range is not None:
            ranges.append(min_range)
        else:
            ranges.append(np.nan)

    return {
        'angles': angles,
        'ranges': np.array(ranges),
        'intensities': np.ones_like(ranges)
    }
