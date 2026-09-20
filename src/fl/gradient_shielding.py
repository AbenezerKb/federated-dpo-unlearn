"""FUGAS-style gradient shielding for server-side aggregation.

Projects unlearning client gradients onto a compatibility subspace
derived from retained clients' gradients, ensuring directional
coherence and non-increasing risk on retained tasks.
"""

import logging
from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_compatibility_subspace(
    retained_deltas: list[NDArray],
    subspace_dim: int = 64,
) -> NDArray:
    """Compute compatibility subspace from retained clients' parameter deltas.

    Uses SVD to find the principal directions of retained client updates.
    The unlearning update will be projected onto this subspace to avoid
    destructive interference.

    Args:
        retained_deltas: List of flattened parameter deltas from retained clients
        subspace_dim: Number of principal components to keep

    Returns:
        Projection matrix (d, subspace_dim) defining the compatibility subspace
    """
    if not retained_deltas:
        raise ValueError("Need at least one retained client delta")

    # Stack deltas: (num_retained, d)
    delta_matrix = np.stack(retained_deltas, axis=0)

    # Center
    mean_delta = delta_matrix.mean(axis=0)
    centered = delta_matrix - mean_delta

    # SVD to find principal directions
    # Only compute top-k singular vectors
    k = min(subspace_dim, min(centered.shape) - 1)
    if k <= 0:
        # Fallback: use mean direction
        norm = np.linalg.norm(mean_delta)
        if norm > 0:
            return (mean_delta / norm).reshape(-1, 1)
        return np.eye(len(mean_delta), 1)

    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Vt rows are right singular vectors (directions in param space)
        basis = Vt[:k].T  # (d, k)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed, using identity projection")
        d = centered.shape[1]
        basis = np.eye(d, min(k, d))

    logger.info(
        f"Compatibility subspace: {basis.shape[1]} dimensions, "
        f"explained variance ratio: {(S[:k]**2).sum() / (S**2).sum():.3f}"
    )

    return basis


def project_onto_subspace(
    delta: NDArray,
    basis: NDArray,
    strength: float = 1.0,
) -> NDArray:
    """Project a parameter delta onto the compatibility subspace.

    Args:
        delta: Flattened parameter delta to project (d,)
        basis: Subspace basis from compute_compatibility_subspace (d, k)
        strength: Interpolation between original (0.0) and projected (1.0)

    Returns:
        Projected delta (d,)
    """
    # Project: delta_proj = basis @ basis.T @ delta
    coords = basis.T @ delta       # (k,)
    projected = basis @ coords     # (d,)

    # Interpolate
    if strength < 1.0:
        projected = strength * projected + (1.0 - strength) * delta

    return projected


def check_directional_coherence(
    unlearn_delta: NDArray,
    retained_deltas: list[NDArray],
) -> dict[str, float]:
    """Check if unlearning delta is directionally coherent with retained updates.

    Returns cosine similarities between unlearning delta and each retained delta.
    Negative similarity = destructive interference.
    """
    unlearn_norm = np.linalg.norm(unlearn_delta)
    if unlearn_norm == 0:
        return {"mean_cosine": 0.0, "min_cosine": 0.0, "conflict_ratio": 1.0}

    cosines = []
    for rd in retained_deltas:
        rd_norm = np.linalg.norm(rd)
        if rd_norm == 0:
            cosines.append(0.0)
            continue
        cos = np.dot(unlearn_delta, rd) / (unlearn_norm * rd_norm)
        cosines.append(float(cos))

    conflicts = sum(1 for c in cosines if c < 0)
    return {
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
        "conflict_ratio": conflicts / len(cosines),
        "per_client_cosines": cosines,
    }


def shield_gradients(
    unlearn_delta: NDArray,
    retained_deltas: list[NDArray],
    subspace_dim: int = 64,
    projection_strength: float = 1.0,
) -> tuple[NDArray, dict]:
    """Full FUGAS gradient shielding pipeline.

    1. Compute compatibility subspace from retained client deltas
    2. Check directional coherence (pre-projection)
    3. Project unlearning delta onto compatible subspace
    4. Check directional coherence (post-projection)

    Args:
        unlearn_delta: Unlearning client's parameter delta
        retained_deltas: Retained clients' parameter deltas
        subspace_dim: Dimensions for compatibility subspace
        projection_strength: How strongly to project (0=no change, 1=full projection)

    Returns:
        (shielded_delta, metrics_dict)
    """
    # Pre-projection coherence
    pre_metrics = check_directional_coherence(unlearn_delta, retained_deltas)
    logger.info(
        f"Pre-shielding: mean_cosine={pre_metrics['mean_cosine']:.4f}, "
        f"conflict_ratio={pre_metrics['conflict_ratio']:.2f}"
    )

    # Compute subspace
    basis = compute_compatibility_subspace(retained_deltas, subspace_dim)

    # Project
    shielded = project_onto_subspace(unlearn_delta, basis, projection_strength)

    # Post-projection coherence
    post_metrics = check_directional_coherence(shielded, retained_deltas)
    logger.info(
        f"Post-shielding: mean_cosine={post_metrics['mean_cosine']:.4f}, "
        f"conflict_ratio={post_metrics['conflict_ratio']:.2f}"
    )

    metrics = {
        "pre_shielding": pre_metrics,
        "post_shielding": post_metrics,
        "subspace_dim": basis.shape[1],
        "delta_norm_before": float(np.linalg.norm(unlearn_delta)),
        "delta_norm_after": float(np.linalg.norm(shielded)),
    }

    return shielded, metrics
