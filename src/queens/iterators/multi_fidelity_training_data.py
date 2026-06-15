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
"""Iterator for generating multi-fidelity training data."""

# pylint: disable=invalid-name
import logging

import numpy as np

from queens.iterators._iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results
from queens.utils.sobol_sequence import sample_sobol_sequence

_logger = logging.getLogger(__name__)


class MultiFidelityTrainingData(Iterator):
    """Multi-fidelity training data iterator.

    Iterator to generate training data to learn the multi-fidelity conditional for BMFIA. Here, we determine optimal training
    points *X_train* and evaluate the low- and high-fidelity model for these
    training inputs, to yield *Y_LF_train* and *Y_HF_train* training data. The
    actual inverse problem is not solved in this module.

    Attributes:
        X_train (np.array): Input training matrix for HF and LF model.
        Y_LF_train (np.array): Corresponding LF model response to *X_train* input.
        Y_HF_train (np.array): Corresponding HF model response to *X_train* input.
        Z_train (np.array): Corresponding LF informative features to *X_train* input.
        hf_model (obj): High-fidelity model object.
        lf_model (obj): Low-fidelity model object.
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        global_settings,
        hf_model,
        lf_model,
        initial_design,
    ):
        """Instantiate the MultiFidelityMapping iterator.

        Args:
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): Settings of the QUEENS experiment including its name
                and the output directory
            hf_model (obj): High-fidelity model object.
            lf_model (obj): Low-fidelity model object.
            initial_design (dict): Dictionary describing initial design.
        """
        super().__init__(None, parameters, global_settings)  # Input prescribed by iterator.py

        # ---------- calculate the initial training samples via classmethods ----------
        x_train = self.calculate_initial_x_train(initial_design, parameters)

        self.X_train = x_train
        self.Y_LF_train = None
        self.Y_HF_train = None
        self.hf_model = hf_model
        self.lf_model = lf_model
        self.output = {}

    @classmethod
    def calculate_initial_x_train(cls, initial_design_dict, parameters):
        """Optimal training data set for probabilistic model.

        Based on the selected design method, determine the optimal set of
        input points X_train to run the HF and the LF model on for the
        construction of the probabilistic surrogate.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)
            parameters (obj): Parameters object

        Returns:
            x_train (np.array): Optimal training input samples
        """
        run_design_method = cls.get_design_method(initial_design_dict)
        x_train = run_design_method(initial_design_dict, parameters)
        return x_train

    @classmethod
    def get_design_method(cls, initial_design_dict):
        """Get the design method for initial training data.

        Select the method for the generation of the initial training data
        for the probabilistic regression model.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.

        Returns:
            run_design_method (obj): Design method for selecting the HF training set
        """
        # check correct inputs
        assert isinstance(
            initial_design_dict, dict
        ), "Input argument 'initial_design_dict' must be of type 'dict'! Abort..."

        assert (
            "type" in initial_design_dict.keys()
        ), "No key 'type' found in 'initial_design_dict'. Abort..."

        # choose design method
        if initial_design_dict["type"] == "random":
            run_design_method = cls.random_design
        elif initial_design_dict["type"] == "sobol":
            run_design_method = cls._sobol_design
        else:
            raise NotImplementedError(
                "The design type you chose for selecting training data is not valid! "
                f"You chose {initial_design_dict['type']} but the only valid options "
                "is 'random'!"
            )

        return run_design_method

    @staticmethod
    def random_design(initial_design_dict, parameters):
        """Generate a uniformly random design strategy.

        Get a random initial design using the Monte-Carlo sampler with a uniform distribution.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)
            parameters (obj): Parameters object

        Returns:
            x_train (np.array): Optimal training input samples
        """
        seed = initial_design_dict["seed"]
        num_samples = initial_design_dict["num_HF_eval"]
        np.random.seed(seed)
        x_train = parameters.draw_samples(num_samples)
        return x_train

    @staticmethod
    def _sobol_design(initial_design_dict, parameters):
        """Generate quasi random design using the Sobol sequence.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)
            parameters (obj): Parameters object

        Returns:
            x_train (np.array): Training input samples from Sobol sequence
        """
        # remove the first point of the Sobol sequence which is zero and not a good training point
        x_train = sample_sobol_sequence(
            dimension=parameters.num_parameters,
            number_of_samples=initial_design_dict["num_HF_eval"] + 1,
            parameters=parameters,
            randomize=False,
            seed=initial_design_dict["seed"],
        )
        return x_train[1:]

    # ----------- main methods of the object from here ----------------------------------------
    def core_run(self):
        """Trigger main or core run of the MultiFidelityMapping iterator.

        It summarizes the actual evaluation of the HF and LF models for these data and the
        determination of LF informative features.

        Returns:
            Z_train (np.array): Matrix with low-fidelity feature training data
            Y_HF_train (np.array): Matrix with HF training data
        """
        # ----- build model on training points and evaluate it -----------------------
        self.eval_model()

        return self.X_train, self.Y_LF_train, self.Y_HF_train

    def expand_training_data(self, additional_x_train, additional_y_lf_train=None):
        """Update or expand the training data.

        Data is appended by an additional input/output vector of data.

        Args:
            additional_x_train (np.array): Additional input vector
            additional_y_lf_train (np.array, optional): Additional LF model response corresponding
                                                        to additional input vector. Default to None

        Returns:
            X_train (np.array): Training input samples
            Y_LF_train (np.array): Matrix with low-fidelity training data
            Y_HF_train (np.array): Matrix with high-fidelity training data
        """
        if additional_y_lf_train is None:
            _logger.info("Starting to compute additional Y_LF_train...")
            additional_y_lf_train = self.lf_model.evaluate(additional_x_train)["result"]
            _logger.info("Additional Y_LF_train were successfully computed!")

        _logger.info("Starting to compute additional Y_LF_train...")
        additional_y_hf_train = self.hf_model.evaluate(additional_x_train)["result"]
        _logger.info("Additional Y_HF_train were successfully computed!")

        self.X_train = np.vstack((self.X_train, additional_x_train))
        self.Y_LF_train = np.vstack((self.Y_LF_train, additional_y_lf_train))
        self.Y_HF_train = np.vstack((self.Y_HF_train, additional_y_hf_train))
        _logger.info("Training data was successfully expanded!")

        return self.X_train, self.Y_LF_train, self.Y_HF_train

    def evaluate_LF_model_for_X_train(self):
        """Evaluate the low-fidelity model for the X_train input data-set."""
        self.Y_LF_train = self.lf_model.evaluate(self.X_train)["result"]

    def evaluate_HF_model_for_X_train(self):
        """Evaluate the high-fidelity model for the X_train input data-set."""
        self.Y_HF_train = self.hf_model.evaluate(self.X_train)["result"]

    def eval_model(self):
        """Evaluate the LF and HF model to for the training inputs.

        *X_train*.
        """
        # ---- run LF model on X_train (potentially we need to iterate over this and the previous
        # step to determine optimal X_train; for now just one sequence)
        _logger.info("-------------------------------------------------------------------")
        _logger.info("Starting to evaluate the low-fidelity model for training points....")
        _logger.info("-------------------------------------------------------------------")

        self.evaluate_LF_model_for_X_train()

        _logger.info("-------------------------------------------------------------------")
        _logger.info("Successfully calculated the low-fidelity training points!")
        _logger.info("-------------------------------------------------------------------")

        # ---- run HF model on X_train
        _logger.info("-------------------------------------------------------------------")
        _logger.info("Starting to evaluate the high-fidelity model for training points...")
        _logger.info("-------------------------------------------------------------------")

        self.evaluate_HF_model_for_X_train()

        _logger.info("-------------------------------------------------------------------")
        _logger.info("Successfully calculated the high-fidelity training points!")
        _logger.info("-------------------------------------------------------------------")

    def post_run(self):
        """Post-run method for the MultiFidelityMapping iterator.

        Here, we determine the LF informative features based on the selected feature strategy.
        """
        output = {
            "X_train": self.X_train,
            "Y_LF_train": self.Y_LF_train,
            "Y_HF_train": self.Y_HF_train,
        }
        write_results(
            output,
            self.global_settings.result_file(extension=".pickle", suffix="_mf_training_data"),
        )
