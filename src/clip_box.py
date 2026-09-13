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


def half_volume_box(bounds):
    """Shrink around the existing center; retain half the original volume."""
    bounds = validate_box(bounds)
    center = bounds.mean(axis=0)
    half_extent = (bounds[1] - bounds[0]) * (0.5 * np.cbrt(0.5))
    return validate_box(np.stack([center - half_extent, center + half_extent]))


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


def fit_subject_box(points):
    """Fit completed geometry around its largest connected dense region.

    Density guides estimation only; no rendering mask is returned. Duplicate
    centers do not inflate support, and sparse background cannot set the center.
    """
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=np.float64)
    points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
    if len(points) < 17:
        return estimate_oriented_box(points)
    distances, _ = cKDTree(points).query(points, k=17)
    radius = float(np.quantile(distances[:, -1], 0.6))
    support = points[distances[:, -1] <= radius * (1 + 1e-10)]
    if radius <= 0 or not len(support):
        return estimate_oriented_box(points)
    cells, inverse, counts = np.unique(
        np.floor((support - support.min(axis=0)) / radius).astype(np.int64),
        axis=0, return_inverse=True, return_counts=True,
    )
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    labels = np.full(len(cells), -1, dtype=int)
    masses = []
    offsets = [offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0)]
    for seed in range(len(cells)):
        if labels[seed] >= 0:
            continue
        label = len(masses)
        labels[seed] = label
        stack, mass = [seed], 0
        while stack:
            i = stack.pop()
            mass += int(counts[i])
            x, y, z = cells[i]
            for dx, dy, dz in offsets:
                j = lookup.get((x + dx, y + dy, z + dz))
                if j is not None and labels[j] < 0:
                    labels[j] = label
                    stack.append(j)
        masses.append(mass)
    subject = support[labels[inverse] == np.argmax(masses)]
    _, axes = estimate_oriented_box(subject)
    local = subject @ axes
    center = np.median(local, axis=0)
    # Size symmetrically about the subject center. Do not halve afterward:
    # that previously cropped the very region used to estimate the box.
    half = max(float(np.quantile(np.abs(local - center), 0.99, axis=0).max()) * 1.05, 1e-3)
    return validate_box(np.stack([center - half, center + half])), axes
