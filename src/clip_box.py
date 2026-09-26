"""Scene bounds and a hard screen-space silhouette for an model-aligned cube."""
from itertools import product
import cv2
import numpy as np


def estimate_box(points):
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError('Cannot estimate clipping bounds without finite points')
    lo, hi = np.quantile(points, [0.01, 0.99], axis=0)
    center = (lo + hi) / 2
    half = max(float((hi - lo).max()) * 0.52, 1e-3)
    return np.stack([center - half, center + half]).astype(np.float32)


def estimate_oriented_box(points):
    """Fit a fixed PCA frame to the central cloud, then bound it in that frame.

    Columns of axes map box-local coordinates into world coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError('Cannot estimate clipping bounds without finite points')
    center = np.median(points, axis=0)
    distance = np.linalg.norm(points - center, axis=1)
    cutoff = np.quantile(distance, 0.98)
    # Include equal-distance boundary points despite floating-point roundoff.
    core = points[distance <= cutoff + max(1., cutoff) * 1e-10]
    axes = np.eye(3)
    if len(core) >= 3:
        covariance = np.cov(core - core.mean(axis=0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] > 1e-12:
            axes = vectors[:, ::-1].copy()
            for i in range(3):
                if axes[np.argmax(np.abs(axes[:, i])), i] < 0:
                    axes[:, i] *= -1
            if np.linalg.det(axes) < 0:
                axes[:, 2] *= -1
    axes = axes.astype(np.float32)
    return estimate_box(points @ axes), axes


def validate_axes(axes):
    axes = np.asarray(axes, dtype=np.float32)
    if (axes.shape != (3, 3) or not np.isfinite(axes).all()
            or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(axes), 1., atol=1e-5)):
        raise ValueError('Clipping axes must be a proper orthonormal rotation')
    return axes


def box_view(view, axes):
    """Camera transform for coordinates in the rotated box frame."""
    result = np.array(view, copy=True)
    result[:3, :3] = view[:3, :3] @ validate_axes(axes)
    return result


def validate_box(bounds):
    bounds = np.asarray(bounds, dtype=np.float32)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError('Clipping bounds must be finite min/max XYZ coordinates')
    return bounds


def box_mask(bounds, view, K, width, height):
    """Pixels whose forward camera ray intersects the cube (including inside views)."""
    bounds = validate_box(bounds)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    rays = np.stack([(x + 0.5 - K[0, 2]) / K[0, 0],
                     (y + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(x)], axis=-1)
    rays = rays @ view[:3, :3]
    origin = -view[:3, :3].T @ view[:3, 3]
    near = np.zeros((height, width), dtype=np.float32)
    far = np.full((height, width), np.inf, dtype=np.float32)
    for axis in range(3):
        direction = rays[..., axis]
        parallel = np.abs(direction) < 1e-8
        safe = np.where(parallel, 1., direction)
        a = (bounds[0, axis] - origin[axis]) / safe
        b = (bounds[1, axis] - origin[axis]) / safe
        lower, upper = np.minimum(a, b), np.maximum(a, b)
        inside = bounds[0, axis] <= origin[axis] <= bounds[1, axis]
        lower = np.where(parallel, -np.inf if inside else np.inf, lower)
        upper = np.where(parallel, np.inf if inside else -np.inf, upper)
        near = np.maximum(near, lower)
        far = np.minimum(far, upper)
    return (far >= near) & (far > 0)


def draw_box(frame, bounds, view, K):
    corners = np.array(list(product(*zip(bounds[0], bounds[1]))), dtype=np.float32)
    camera = corners @ view[:3, :3].T + view[:3, 3]
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j <= i:
                continue
            a, b = camera[i].copy(), camera[j].copy()
            if max(a[2], b[2]) < 1e-3:
                continue
            if a[2] < 1e-3:
                a += (b - a) * ((1e-3 - a[2]) / (b[2] - a[2]))
            if b[2] < 1e-3:
                b += (a - b) * ((1e-3 - b[2]) / (a[2] - b[2]))
            pa, pb = K @ a, K @ b
            pa = tuple(np.clip(pa[:2] / pa[2], -1e8, 1e8).astype(int))
            pb = tuple(np.clip(pb[:2] / pb[2], -1e8, 1e8).astype(int))
            ok, pa, pb = cv2.clipLine((0, 0, frame.shape[1], frame.shape[0]), pa, pb)
            if ok:
                cv2.line(frame, pa, pb, (80, 220, 255), 1, cv2.LINE_AA)


def fit_scene_box(points, opacities=None, grid=192, min_share=0.01, margin=0.03):
    """Box around everything connected to the reconstructed scene.

    Gaussians are binned on an occupancy grid (at most `grid` cells along the
    longest side, weighted by opacity) and grouped into connected regions. Every
    region holding at least `min_share` of the total is kept, so thin structures
    joined to the scene (towers, masts, bridges) stay inside, while isolated
    floaters are left out. The box is aligned with the scene's principal axes and
    sized per axis, so wide, flat drone scenes are not forced into a cube.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=np.float64)
    weights = (np.ones(len(points)) if opacities is None
               else np.clip(np.asarray(opacities, dtype=np.float64).reshape(-1), 0, 1))
    valid = np.isfinite(points).all(axis=1) & np.isfinite(weights)
    visible = valid & (weights > 0.02)  # near-transparent splats do not set the bounds
    if visible.sum() >= 16:
        valid = visible
    points, weights = points[valid], np.maximum(weights[valid], 1e-6)
    _, axes = estimate_oriented_box(points)  # raises for an empty cloud
    local = points @ axes.astype(np.float64)
    if len(points) < 16:  # too few to judge connectivity: keep every point
        low, high = local.min(axis=0), local.max(axis=0)
        pad = np.maximum((high - low) * margin, 1e-3)
        return validate_box(np.stack([low - pad, high + pad])), axes
    # Grid over the robust extent plus a wide border; anything beyond is a far outlier.
    lo, hi = np.quantile(local, [0.001, 0.999], axis=0)
    lo, hi = lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.25
    # Cells at least twice the typical point spacing, so a surface is one connected
    # region even when the cloud is sparse (e.g. COLMAP points before training).
    sample = local if len(local) <= 50000 else local[np.random.default_rng(0).choice(len(local), 50000, replace=False)]
    sample = np.unique(sample, axis=0)  # repeated positions must not read as zero spacing
    spacing = float(np.median(cKDTree(sample).query(sample, k=min(4, len(sample)))[0][:, 1:].mean(axis=1)))
    cell = max(float((hi - lo).max()) / grid, 2 * spacing, 1e-6)
    shape = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    index = np.floor((local - lo) / cell).astype(int)
    inside = ((index >= 0) & (index < shape)).all(axis=1)
    occupancy = np.zeros(shape, dtype=np.float64)
    np.add.at(occupancy, tuple(index[inside].T), weights[inside])
    labels, count = ndimage.label(occupancy > 0, structure=np.ones((3, 3, 3)))
    if count == 0:
        return estimate_box(local), axes
    mass = np.asarray(ndimage.sum(occupancy, labels, index=np.arange(1, count + 1)))
    regions = np.flatnonzero((mass >= min_share * mass.sum()) | (mass == mass.max())) + 1
    kept = inside.copy()
    kept[inside] = np.isin(labels[tuple(index[inside].T)], regions)
    selected = local[kept]
    low, high = np.quantile(selected, [0.0005, 0.9995], axis=0)
    pad = (high - low) * margin + cell
    return validate_box(np.stack([low - pad, high + pad])), axes
