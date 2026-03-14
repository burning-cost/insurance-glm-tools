"""
Min-exposure, min-claims, and monotonicity constraints for merged factor groups.

After fused lasso fusion, some groups may contain very few exposure units or
claims — too little credibility to support a standalone relativity. This module
absorbs under-credible groups into their nearest-coefficient neighbour, and
provides isotonic regression to enforce monotone pricing factors.

Design choice: nearest-coefficient neighbour (not nearest-level neighbour).
This is conservative: a tiny group gets absorbed into whichever group it
already looks most like, rather than the adjacent one, which may be far away.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def enforce_min_exposure(
    groups: NDArray[np.int32],
    exposure_by_level: NDArray[np.float64],
    coefficients_by_level: NDArray[np.float64],
    min_exposure: float,
) -> NDArray[np.int32]:
    """
    Iteratively absorb under-credible groups into nearest-coefficient neighbour.

    Parameters
    ----------
    groups : NDArray[np.int32]
        Group label for each level (length K). Labels need not be contiguous.
    exposure_by_level : NDArray[np.float64]
        Total exposure for each level (length K).
    coefficients_by_level : NDArray[np.float64]
        Coefficient (β) for each level (length K), used to find nearest neighbour.
    min_exposure : float
        Minimum total exposure required for a group to stand on its own.

    Returns
    -------
    NDArray[np.int32]
        Updated group labels after absorption (length K).
    """
    if min_exposure <= 0:
        return groups.copy()

    groups = groups.copy()

    # Iterate until no further merges are needed
    for _ in range(len(groups)):  # worst case: K-1 merges
        # Compute group-level stats
        unique_groups = np.unique(groups)
        if len(unique_groups) == 1:
            break

        group_exposure: dict[int, float] = {}
        group_coef: dict[int, float] = {}

        for g in unique_groups:
            mask = groups == g
            group_exposure[g] = float(exposure_by_level[mask].sum())
            group_coef[g] = float(np.average(
                coefficients_by_level[mask],
                weights=exposure_by_level[mask] if exposure_by_level[mask].sum() > 0 else None,
            ))

        # Find the smallest under-exposure group
        under = {g: e for g, e in group_exposure.items() if e < min_exposure}
        if not under:
            break

        # Take the group with least exposure
        target_g = min(under, key=lambda g: under[g])
        target_coef = group_coef[target_g]

        # Find nearest-coefficient neighbour (different group)
        other_groups = [g for g in unique_groups if g != target_g]
        nearest = min(other_groups, key=lambda g: abs(group_coef[g] - target_coef))

        # Absorb target into nearest — relabel all target levels as nearest
        groups[groups == target_g] = nearest

    return groups


def enforce_min_claims(
    groups: NDArray[np.int32],
    claim_counts: NDArray[np.float64],
    coefficients_by_level: NDArray[np.float64],
    min_claims: int,
) -> NDArray[np.int32]:
    """
    Merge groups whose total claim count falls below the minimum threshold.

    Same greedy nearest-coefficient absorption as enforce_min_exposure, but
    uses claim counts as the credibility measure rather than exposure. Useful
    when claim count is the binding constraint (e.g. for low-frequency perils).

    Parameters
    ----------
    groups : NDArray[np.int32]
        Group label for each level (length K).
    claim_counts : NDArray[np.float64]
        Claim count for each level (length K).
    coefficients_by_level : NDArray[np.float64]
        Coefficient (β) for each level (length K), used to find nearest neighbour.
    min_claims : int
        Minimum claims per merged group.

    Returns
    -------
    NDArray[np.int32]
        Updated group labels after absorption (length K).
    """
    return enforce_min_exposure(
        groups,
        claim_counts.astype(np.float64),
        coefficients_by_level,
        float(min_claims),
    )


def _pav_increasing(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Pool adjacent violators for monotone increasing constraint.

    Pure NumPy implementation that works with any scipy version.

    Parameters
    ----------
    values : NDArray[np.float64]
        Input values to project onto the increasing cone.

    Returns
    -------
    NDArray[np.float64]
        Monotone increasing values of the same length.
    """
    result = values.copy()

    # Build blocks iteratively
    blocks: list[list[float]] = []
    block_means: list[float] = []

    for v in result:
        block = [float(v)]
        mean = float(v)
        while block_means and block_means[-1] > mean:
            # Merge with previous block
            prev = blocks.pop()
            block = prev + block
            mean = sum(block) / len(block)
            block_means.pop()
        blocks.append(block)
        block_means.append(mean)

    # Flatten
    idx = 0
    for block, mean in zip(blocks, block_means):
        for _ in block:
            result[idx] = mean
            idx += 1

    return result


def enforce_monotonicity(
    group_coefficients: pd.Series,
    direction: str = "increasing",
) -> pd.Series:
    """
    Project group coefficients onto the monotone cone.

    Uses scipy's isotonic_regression (scipy >= 1.12) when available,
    falling back to a pure NumPy pool adjacent violators implementation
    for older scipy versions.

    Parameters
    ----------
    group_coefficients : pd.Series
        Coefficients indexed by group code (integer). Groups should be in
        the natural ordering (0, 1, 2, ...).
    direction : str
        'increasing' or 'decreasing'.

    Returns
    -------
    pd.Series
        Monotone coefficients with the same index as the input.

    Raises
    ------
    ValueError
        If direction is not 'increasing' or 'decreasing'.
    """
    if direction not in ("increasing", "decreasing"):
        raise ValueError(
            f"direction must be 'increasing' or 'decreasing', got '{direction}'."
        )

    sorted_idx = group_coefficients.sort_index()
    values = sorted_idx.values.astype(np.float64)

    if direction == "decreasing":
        values = -values

    # Try scipy >= 1.12 first
    try:
        from scipy.optimize import isotonic_regression
        result = isotonic_regression(values, increasing=True)
        if hasattr(result, "x"):
            monotone_values = result.x
        else:
            monotone_values = np.asarray(result)
    except ImportError:
        # Fallback: pure NumPy PAV
        monotone_values = _pav_increasing(values)

    if direction == "decreasing":
        monotone_values = -monotone_values

    return pd.Series(monotone_values, index=sorted_idx.index)


def check_monotonicity(
    group_coefficients: pd.Series,
    direction: str = "increasing",
    tol: float = 1e-6,
) -> tuple[bool, list[int]]:
    """
    Check whether group coefficients satisfy a monotonicity constraint.

    Parameters
    ----------
    group_coefficients : pd.Series
        Coefficients indexed by group code.
    direction : str
        'increasing' or 'decreasing'.
    tol : float
        Tolerance for numerical noise.

    Returns
    -------
    bool
        True if monotone.
    list[int]
        Indices of violating pairs (group codes where the constraint is broken).

    Raises
    ------
    ValueError
        If direction is not 'increasing' or 'decreasing'.
    """
    if direction not in ("increasing", "decreasing"):
        raise ValueError(
            f"direction must be 'increasing' or 'decreasing', got '{direction}'."
        )

    sorted_coef = group_coefficients.sort_index()
    values = sorted_coef.values
    indices = sorted_coef.index.tolist()
    violations: list[int] = []

    for i in range(len(values) - 1):
        if direction == "increasing" and values[i] > values[i + 1] + tol:
            violations.append(indices[i])
        elif direction == "decreasing" and values[i] < values[i + 1] - tol:
            violations.append(indices[i])

    return len(violations) == 0, violations


def compute_group_exposure(
    groups: NDArray[np.int32],
    exposure_by_level: NDArray[np.float64],
) -> dict[int, float]:
    """
    Compute total exposure per group.

    Parameters
    ----------
    groups : NDArray[np.int32]
        Group label for each level.
    exposure_by_level : NDArray[np.float64]
        Exposure for each level.

    Returns
    -------
    dict[int, float]
        Mapping from group label to total exposure.
    """
    result: dict[int, float] = {}
    for g in np.unique(groups):
        result[int(g)] = float(exposure_by_level[groups == g].sum())
    return result


def relabel_groups_contiguous(groups: NDArray[np.int32]) -> NDArray[np.int32]:
    """
    Relabel group integers to be contiguous starting from 0.

    After absorption, group labels may have gaps (e.g. 0, 2, 5).
    This returns labels 0, 1, 2 in order of first appearance.

    Parameters
    ----------
    groups : NDArray[np.int32]
        Group labels, possibly non-contiguous.

    Returns
    -------
    NDArray[np.int32]
        Contiguous group labels preserving order of first occurrence.
    """
    seen: dict[int, int] = {}
    counter = 0
    new_groups = np.empty_like(groups)
    for i, g in enumerate(groups):
        if g not in seen:
            seen[g] = counter
            counter += 1
        new_groups[i] = seen[g]
    return new_groups
