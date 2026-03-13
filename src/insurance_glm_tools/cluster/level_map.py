"""
LevelMap: the output artefact for each clustered factor.

A LevelMap records the mapping from original factor levels to merged groups,
together with group-level statistics (coefficient, exposure). This is the
interface between the algorithm and downstream model building.

Design: immutable after construction. Convert to DataFrame for inspection,
use apply() to recode a new series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LevelMap:
    """
    Mapping from original factor levels to merged groups for one factor.

    Attributes
    ----------
    factor : str
        Name of the original factor column.
    ordered_levels : tuple
        All original levels in sorted order.
    groups : tuple[int, ...]
        Group label for each level (same length as ordered_levels).
    coefficients : tuple[float, ...]
        Log-relativity coefficient for each group (length = n_groups).
        Index by group label.
    group_exposure : tuple[float, ...]
        Total exposure for each group (length = n_groups).
    """

    factor: str
    ordered_levels: tuple
    groups: tuple
    coefficients: tuple
    group_exposure: tuple

    @property
    def n_levels(self) -> int:
        """Number of original levels."""
        return len(self.ordered_levels)

    @property
    def n_groups(self) -> int:
        """Number of distinct merged groups."""
        return len(self.coefficients)

    @property
    def level_to_group(self) -> dict:
        """Dict mapping each original level to its group label."""
        return dict(zip(self.ordered_levels, self.groups))

    def apply(self, series: pd.Series) -> pd.Series:
        """
        Recode a series of original levels to merged group labels.

        Parameters
        ----------
        series : pd.Series
            Series of original factor values.

        Returns
        -------
        pd.Series
            Series of integer group labels.
        """
        mapping = self.level_to_group
        return series.map(mapping)

    def to_df(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame with one row per original level.

        Columns: original_level, merged_group, coefficient, group_exposure.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        ltg = self.level_to_group
        for level in self.ordered_levels:
            g = ltg[level]
            rows.append({
                "original_level": level,
                "merged_group": g,
                "coefficient": self.coefficients[g] if g < len(self.coefficients) else np.nan,
                "group_exposure": self.group_exposure[g] if g < len(self.group_exposure) else np.nan,
            })
        return pd.DataFrame(rows)

    def group_summary(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame with one row per merged group.

        Columns: merged_group, levels, coefficient, group_exposure.

        Returns
        -------
        pd.DataFrame
        """
        ltg = self.level_to_group
        # Collect levels per group
        group_levels: dict[int, list] = {}
        for level, g in ltg.items():
            group_levels.setdefault(g, []).append(level)

        rows = []
        for g in sorted(group_levels):
            rows.append({
                "merged_group": g,
                "levels": sorted(group_levels[g]),
                "coefficient": self.coefficients[g] if g < len(self.coefficients) else np.nan,
                "group_exposure": self.group_exposure[g] if g < len(self.group_exposure) else np.nan,
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"LevelMap(factor='{self.factor}', "
            f"n_levels={self.n_levels}, "
            f"n_groups={self.n_groups})"
        )


def build_level_map(
    factor: str,
    ordered_levels: list,
    groups: np.ndarray,
    coefficients: np.ndarray,
    exposure_by_level: np.ndarray,
) -> LevelMap:
    """
    Construct a LevelMap from raw arrays.

    Parameters
    ----------
    factor : str
        Factor name.
    ordered_levels : list
        All original levels in order.
    groups : np.ndarray
        Group label per level (length K).
    coefficients : np.ndarray
        Coefficient per group, indexed by group label.
    exposure_by_level : np.ndarray
        Exposure per level (length K).

    Returns
    -------
    LevelMap
    """
    unique_groups = sorted(int(g) for g in np.unique(groups))
    n_groups = len(unique_groups)

    # Compute group exposure
    group_exposure_arr = np.zeros(n_groups, dtype=np.float64)
    for g in unique_groups:
        mask = groups == g
        group_exposure_arr[g] = float(exposure_by_level[mask].sum())

    return LevelMap(
        factor=factor,
        ordered_levels=tuple(ordered_levels),
        groups=tuple(int(g) for g in groups),
        coefficients=tuple(float(c) for c in coefficients),
        group_exposure=tuple(float(e) for e in group_exposure_arr),
    )
