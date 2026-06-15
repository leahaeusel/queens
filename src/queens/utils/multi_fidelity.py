#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Utility functions for multi-fidelity analysis."""

from abc import ABC, abstractmethod

import numpy as np


class MultiFidelityFeatureStrategy(ABC):
    """Abstract base class for multi-fidelity feature strategies."""

    @abstractmethod
    def set_feature_strategy(self, y_lf_mat, x_mat):
        """Get the low-fidelity feature matrix."""

    @staticmethod
    def _validate_inputs(y_lf_mat, x_mat=None, coords_mat=None):
        assert (
            y_lf_mat.ndim == 2
        ), f"Dimension of y_lf_mat must be 2 but you provided dim={y_lf_mat.ndim}. Abort..."
        assert (
            x_mat is None or x_mat.ndim == 2
        ), f"Dimension of x_mat must be 2 but you provided dim={x_mat.ndim}. Abort..."
        assert (
            coords_mat is None or coords_mat.ndim == 2
        ), f"Dimension of coords_mat must be 2 but you provided dim={coords_mat.ndim}. Abort..."


class ManualFeatures(MultiFidelityFeatureStrategy):
    """Feature strategy using manually selected input columns."""

    def __init__(self, x_cols):
        super().__init__()
        self.x_cols = x_cols

    def set_feature_strategy(self, y_lf_mat: np.ndarray, x_mat: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            y_lf_mat: num_samples x num_coordinates
            x_mat: num_samples x num_input_features

        Returns:
            Feature matrix of shape (num_selected_features + 1, num_samples, num_coordinates)
        """
        self._validate_inputs(y_lf_mat, x_mat)

        idx_lst = self.x_cols
        assert isinstance(idx_lst, list), "Entries of X_cols must be in list format! Abort..."
        assert (
            idx_lst != []
        ), "The index list for selection of manual features must not be empty!, Abort..."
        gamma_mat = x_mat[:, idx_lst]
        assert (
            gamma_mat.shape[0] == y_lf_mat.shape[0]
        ), "Dimensions of gamma_mat and y_lf_mat do not agree! Abort..."

        z_lst = []
        for y_per_coordinate in y_lf_mat.T:
            z_lst.append(np.hstack([y_per_coordinate.reshape(-1, 1), gamma_mat]))

        z_mat = np.array(z_lst).squeeze().T
        assert z_mat.ndim == 3, "z_mat should be a 3d tensor if man features are used! Abort..."
        return z_mat


class CoordinateFeatures(MultiFidelityFeatureStrategy):
    """Feature strategy using coordinate columns."""

    def __init__(self, coord_cols, coords_mat):
        super().__init__()

        assert isinstance(
            coord_cols, list
        ), "Entries of coord_cols must be in list format! Abort..."
        assert (
            coord_cols != []
        ), "The index list for selection of manual features must not be empty!, Abort..."

        self.coords_mat = coords_mat
        self.coord_features = coords_mat[:, coord_cols]

    def set_feature_strategy(self, y_lf_mat, x_mat):
        self._validate_inputs(y_lf_mat, coords_mat=self.coords_mat)

        assert (
            self.coord_features.shape[0] == y_lf_mat.shape[0]
        ), "Dimensions of coord_features and y_lf_mat do not agree! Abort..."

        z_lst = []
        for y_per_coordinate in y_lf_mat.T:
            z_lst.append(np.hstack([y_per_coordinate.reshape(-1, 1), self.coord_features]))

        z_mat = np.array(z_lst).squeeze().T
        assert z_mat.ndim == 3, "z_mat should be a 3d tensor if coord_features are used! Abort..."
        breakpoint()
        return z_mat


class NoFeatures(MultiFidelityFeatureStrategy):
    """Feature strategy using only low-fidelity outputs."""

    def set_feature_strategy(self, y_lf_mat, x_mat):
        """_summary_

        Args:
            y_lf_mat: num_samples x num_coordinates
            x_mat: num_samples x num_input_features

        Returns:
            Feature matrix of shape (1, num_samples, num_coordinates)
        """
        self._validate_inputs(y_lf_mat)
        return y_lf_mat[None, :, :]


class TimeFeatures(MultiFidelityFeatureStrategy):
    """Feature strategy using time features."""

    def __init__(self, time_vec):
        super().__init__()
        self.time_vec = time_vec

    def set_feature_strategy(self, y_lf_mat, x_mat):
        self._validate_inputs(y_lf_mat)

        time_repeat = int(y_lf_mat.shape[0] / self.time_vec.size)
        time_vec = np.repeat(self.time_vec.reshape(-1, 1), repeats=time_repeat, axis=0)

        z_mat = np.hstack([y_lf_mat, time_vec])
        breakpoint()
        return z_mat
