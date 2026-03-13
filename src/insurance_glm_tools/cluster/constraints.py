"""
Min-exposure constraint enforcement for merged factor groups.

After fused lasso fusion, some groups may contain very few exposure units —
too little credibility to support a standalone relativty. This module absorbs
under-credible groups into their nearest-coefficient neighbour, iterating
until all groups meet the minimum exposure threshold.

Design choice: nearest-coefficient neighbour (not nearest-level neighbour).
This is conservative: a tiny group gets absorbed into whichever group it
already looks most like, rather than the adjacent one, which may be far away.
"""

from __future__ import annotations

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
