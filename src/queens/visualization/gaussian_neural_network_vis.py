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
"""Plotting functions for the Gaussian Neural Network."""

from pathlib import Path

import plotly.graph_objects as go


def plot_loss(history, loss_plot_path):
    """Plot the loss function over the training epochs.

    Args:
        history (obj): Tensorflow history object of the training routine
        loss_plot_path (str): Path to save the loss plot
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history["loss"], mode="lines", name="Loss"))
    if "val_loss" in history.history:
        fig.add_trace(
            go.Scatter(y=history.history["val_loss"], mode="lines", name="Validation Loss")
        )
    fig.update_layout(xaxis_title="# epochs", yaxis_title="-log lik.")
    loss_plot_path = Path(loss_plot_path)
    loss_plot_path.mkdir(exist_ok=True)
    fig.write_html(loss_plot_path / "loss_plot.html")
