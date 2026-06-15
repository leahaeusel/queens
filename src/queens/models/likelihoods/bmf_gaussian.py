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
"""Multi-fidelity Gaussian likelihood model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from queens.distributions.mean_field_normal import MeanFieldNormal
from queens.models.likelihoods._likelihood import Likelihood
from queens.utils.ascii_art import print_bmfia_acceleration
from queens.utils.logger_settings import log_init_args

if TYPE_CHECKING:
    from queens.models.surrogates.multi_fidelity_conditional import MultiFidelityConditional
    from queens.utils.experimental_data_reader import ExperimentalDataReader
    from queens.utils.multi_fidelity import MultiFidelityFeatureStrategy

_logger = logging.getLogger(__name__)


class MultiFidelityGaussian(Likelihood):
    """Multi fidelity Gaussian likelihood function.

    Multi-fidelity likelihood of the Bayesian multi-fidelity inverse
    analysis scheme [1, 2].

    Attributes:
        coords_mat (np.array): Row-wise coordinates at which the observations were recorded
        normal_distribution (obj): Mean field normal distribution object
        noise_var (np.array): Noise variance of the observations
        likelihood_counter (int): Internal counter for the likelihood evaluation
        num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                                      dependency in refinement step

    References:
        [1] Nitzler, J.; Biehler, J.; Fehn, N.; Koutsourelakis, P.-S. and Wall, W.A. (2020),
            "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
            Propagation in Complex Physical Simulations", arXiv:2001.02892

        [2] Nitzler J.; Wall, W. A. and Koutsourelakis P.-S. (2021),
            "An efficient Bayesian multi-fidelity framework for the solution of high-dimensional
            inverse problems in computationally demanding simulations", unpublished internal report
    """

    @log_init_args
    def __init__(
        self,
        forward_model,
        multi_fidelity_surrogate: MultiFidelityConditional,
        experimental_data_reader: ExperimentalDataReader | None = None,
        y_obs=None,
        coords_mat=None,
        feature_strategy: MultiFidelityFeatureStrategy | None = None,
        noise_value=None,
        num_refinement_samples=None,
    ):
        """Instantiate the multi-fidelity likelihood class.

        Args:
            forward_model (obj): Forward model to iterate; here: the low fidelity model
            multi_fidelity_surrogate (MultiFidelityConditional): Multi-fidelity surrogate model
                that maps the low-fidelity model response to the high-fidelity model response
            experimental_data_reader (obj): Experimental data reader object
            y_obs (array_like): Observations
            coords_mat (array_like): Matrix with coordinate values for observations
            feature_strategy (MultiFidelityFeatureStrategy): Object describing the strategy for the
                selection of informative input features.
            noise_value (array_like): Noise variance of the observations
            num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                dependency in refinement step
        """
        if experimental_data_reader is not None:
            if y_obs is not None or coords_mat is not None:
                _logger.info(
                    "You provided an experimental data reader and y_obs and/or coords_mat. "
                    "In the following, the provided experimental data reader is used and the provided y_obs and coords_mat are ignored."
                )
            y_obs, coords_mat, _ = experimental_data_reader.get_experimental_data()

        # avoid interference with forward model evaluations by the multi-fidelity training iterator
        forward_model = forward_model.copy()

        super().__init__(forward_model, y_obs)

        # ----------------------- initialize the mean field normal distribution ------------------
        noise_variance = np.array(noise_value)
        dimension = y_obs.size

        # build distribution with dummy values; parameters might change during runtime
        mean_field_normal = MeanFieldNormal(
            mean=y_obs, variance=noise_variance, dimension=dimension
        )

        # ---------------------- initialize attributes  ------------------
        self.mf_surrogate = multi_fidelity_surrogate
        self.coords_mat = coords_mat
        self.feature_strategy = feature_strategy
        self.min_log_lik_mf = None
        self.normal_distribution = mean_field_normal
        self.noise_var = noise_variance
        self.likelihood_counter = 1
        self.num_refinement_samples = num_refinement_samples

        _logger.info("---------------------------------------------------------------------")
        _logger.info("Speed-up through Multi-fidelity Likelihood evaluation!")
        _logger.info("---------------------------------------------------------------------")
        print_bmfia_acceleration()

    def _evaluate(self, samples):
        """Evaluate multi-fidelity likelihood.

        Evaluation with current set of variables
        which are an attribute of the underlying low-fidelity simulation model.

        Args:
            samples (np.ndarray): Evaluated samples

        Returns:
            dict: Vector of log-likelihood values per model input.
        """
        # reshape the model output according to the number of coordinates
        num_coordinates = self.coords_mat.shape[0]
        num_samples = samples.shape[0]

        # we explicitly cut the array at the variable size as within one batch several chains
        # e.g., in MCMC might be calculated; we only want the last chain here
        forward_model_output = self.forward_model.evaluate(samples)["result"].reshape(
            -1, num_coordinates
        )[:num_samples, :]

        mf_log_likelihood = self.evaluate_from_output(samples, forward_model_output)
        self.response = {
            "forward_model_output": forward_model_output,
            "mf_log_likelihood": mf_log_likelihood,
        }
        return {"result": mf_log_likelihood}

    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        partial_grad = self.partial_grad_evaluate(samples, self.response["forward_model_output"])
        upstream_gradient = upstream_gradient * partial_grad
        gradient = self.forward_model.grad(samples, upstream_gradient)
        return gradient

    def evaluate_from_output(self, samples, forward_model_output):
        """Evaluate multi-fidelity likelihood from forward model output.

        Args:
            samples (np.ndarray): Samples to evaluate
            forward_model_output (np.ndarray): Forward model output

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        # evaluate the modified multi-fidelity likelihood expression with LF model response
        mf_log_likelihood = self.evaluate_mf_likelihood(samples, forward_model_output)
        self.likelihood_counter += 1
        return mf_log_likelihood

    def partial_grad_evaluate(self, forward_model_input, forward_model_output):
        """Implement the partial derivative of the evaluate method.

        The partial derivative w.r.t. the output of the sub-model is for example
        required to calculate gradients of the current model w.r.t. to the sample
        input.

        Args:
            forward_model_input (np.array): Sample inputs of the model run (here not required).
            forward_model_output (np.array): Output of the underlying sub- or forward model
                                             for the current batch of sample inputs.

        Returns:
            grad_out (np.array): Evaluated partial derivative of the evaluation function
                                 w.r.t. the output of the underlying sub-model.
        """
        # construct LF feature matrix
        z_mat = self.feature_strategy.set_feature_strategy(
            forward_model_output, forward_model_input
        )

        # Get the response matrices of the multi-fidelity mapping
        result = self.mf_surrogate.predict(z_mat, gradient_bool=True)
        m_f_mat = result["result"]
        var_y_mat = result["variance"]
        grad_m_f_mat = result["grad_mean"]
        grad_var_y_mat = result["grad_var"]

        if grad_m_f_mat.ndim == 3:
            grad_m_f_mat = grad_m_f_mat[:, :, 0]  # extract only derivative w.r.t. to LF output
            grad_var_y_mat = grad_var_y_mat[:, :, 0]  # extract only derivative w.r.t. to LF output

        assert np.array_equal(
            m_f_mat.shape[1], np.atleast_2d(self.y_obs).shape[1]
        ), "Column dimension of the probab. regression output and y_obs do not agree!"

        # here we iterate over samples meaning we
        # iterate here over all surrogates simultaneously such that
        # the new m is a vector of all e.g. first entries in all surrogates
        log_lik_mf_lst = []
        grad_log_lik_lst = []
        for m_f_vec, variance_vec, grad_m_f, grad_var_y in zip(
            m_f_mat, var_y_mat, grad_m_f_mat, grad_var_y_mat, strict=True
        ):
            self.normal_distribution.update_variance(
                variance_vec.flatten() + self.noise_var.flatten()
            )
            log_lik_mf_lst.append(self.normal_distribution.logpdf(m_f_vec.reshape(1, -1)))
            grad_log_lik_lst.append(
                self.grad_log_pdf_d_ylf(m_f_vec, grad_m_f, grad_var_y).flatten()
            )

        log_lik_mf_output = np.array(log_lik_mf_lst).reshape(-1, 1)
        grad_out = np.array(grad_log_lik_lst)

        if self.min_log_lik_mf is None:
            self.min_log_lik_mf = np.min(log_lik_mf_output)

        return grad_out

    def evaluate_mf_likelihood(self, x_batch, y_lf_mat):
        """Evaluate the Bayesian multi-fidelity likelihood as described in [1].

        Args:
            x_batch (np.array): Input batch matrix; rows correspond to one input vector;
                                different dimensions along columns

            y_lf_mat (np.array): Response matrix of the low-fidelity model; Row-wise corresponding
                                 to rows in x_batch input batch matrix. Different coordinate
                                 locations along the columns

        Returns:
            log_lik_mf_output (tuple): Tuple with vector of log-likelihood values
                                       per model input and potentially the gradient
                                       of the model w.r.t. its inputs


        References:
            [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
                "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
                Propagation in Complex Physical Simulations", arXiv:2001.02892
        """
        # construct LF feature matrix:
        z_mat = self.feature_strategy.set_feature_strategy(y_lf_mat, x_batch)
        # Get the response matrices of the multi-fidelity mapping
        result = self.mf_surrogate.evaluate(z_mat)
        m_f_mat = result["result"]
        var_y_mat = result["variance"]
        assert np.array_equal(
            m_f_mat.shape[1], np.atleast_2d(self.y_obs).shape[1]
        ), "Column dimension of the probab. regression output and y_obs do not agree! Abort..."

        # iterate here over all surrogates simultaneously such that the
        # new m is a vector of all, e.g., first entries in all surrogates
        log_lik_mf_lst = []
        for m_f_vec, variance_vec in zip(m_f_mat, var_y_mat, strict=True):
            self.normal_distribution.update_variance(
                variance_vec.flatten() + self.noise_var.flatten()
            )
            log_lik_mf_lst.append(self.normal_distribution.logpdf(m_f_vec.reshape(1, -1)))
        log_lik_mf_output = np.array(log_lik_mf_lst).reshape(-1)

        if self.min_log_lik_mf is None:
            self.min_log_lik_mf = np.min(log_lik_mf_output)

        return log_lik_mf_output

    def grad_log_pdf_d_ylf(self, m_f_vec, grad_m_f_dy, grad_var_y_dy):
        """Calculate the gradient of the logpdf w.r.t. to the LF model output.

        The gradient is calculated from the individual partial derivatives
        and then composed in this method.

        Args:
            m_f_vec (np.array): mean vector of the probabilistic surrogate evaluated at sample
                                points
            grad_m_f_dy (np.array): gradient of the mean function/vector of the probabilistic
                                 regression model w.r.t. the regression model's input
            grad_var_y_dy (np.array): gradient of the variance function/vector of the probabilistic
                                   regression model w.r.t. the regression model's input

        Returns:
            d_log_lik_d_z (np.array): gradient of the logpdf w.r.t. y_lf
        """
        d_log_lik_d_m_f = self.normal_distribution.grad_logpdf(m_f_vec).reshape(1, -1)
        d_log_lik_d_var = self.normal_distribution.grad_logpdf_var(m_f_vec).reshape(1, -1)

        d_log_lik_d_y = d_log_lik_d_m_f * grad_m_f_dy.reshape(
            1, -1
        ) + d_log_lik_d_var * grad_var_y_dy.reshape(1, -1)

        return d_log_lik_d_y
