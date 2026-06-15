"""Ensemble model that concatenates outputs from multiple models."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from queens.models._model import Model
from queens.utils.logger_settings import log_init_args


class Ensemble(Model):
    """Ensemble model that evaluates multiple models and concatenates outputs.

    Attributes:
            models (list[Model]): Sub-models evaluated for each sample batch.
    """

    @log_init_args
    def __init__(self, models: Iterable[Model]):
        """Initialize ensemble model.

        Args:
                models (Iterable[Model]): Models to evaluate and concatenate.
        """
        super().__init__()
        self.models = list(models)
        if not self.models:
            raise ValueError("At least one model must be provided.")
        self._output_sizes: list | None = None

    def _evaluate(self, samples: np.ndarray):
        """Evaluate all sub-models and concatenate their results.

        Args:
            samples: Input samples.

        Returns:
            Response with concatenated results.
        """
        results = []
        output_sizes = []

        for model in self.models:
            model_result = np.asarray(model.evaluate(samples)["result"])
            if model_result.ndim == 1:
                model_result = model_result.reshape(-1, 1)
            if model_result.shape[0] != samples.shape[0]:
                raise ValueError(
                    "First dimension of model output must match the number of samples."
                )
            results.append(model_result)
            output_sizes.append(model_result.shape[1])

        self._output_sizes = output_sizes
        self.response = {"result": np.concatenate(results, axis=1)}
        return self.response

    def grad(self, samples: np.ndarray, upstream_gradient: np.ndarray):
        r"""Evaluate gradient of concatenated outputs w.r.t. input samples.

        Args:
            samples: Input samples.
            upstream_gradient: Upstream gradient for concatenated outputs.

        Returns:
            Gradient w.r.t. input samples.
        """
        if self._output_sizes is None:
            raise ValueError("Call evaluate before grad to determine output sizes.")

        if upstream_gradient.ndim == 1:
            upstream_gradient = upstream_gradient.reshape(-1, 1)
        if upstream_gradient.shape[0] != samples.shape[0]:
            raise ValueError("Upstream gradient rows must match the number of samples.")

        total_outputs = sum(self._output_sizes)
        if upstream_gradient.shape[1] != total_outputs:
            raise ValueError("Upstream gradient columns must match concatenated output size.")

        start = 0
        gradient = None
        for model, output_size in zip(self.models, self._output_sizes):
            end = start + output_size
            model_grad = model.grad(samples, upstream_gradient[:, start:end])
            gradient = model_grad if gradient is None else gradient + model_grad
            start = end

        return gradient

    def copy(self):
        new = type(self).__new__(type(self))
        Model.__init__(new)
        new.models = [model.copy() for model in self.models]
        new._output_sizes = self._output_sizes
        return new
