"""TODO_doc."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from queens.example_simulator_functions.gaussian_mixture_logpdf import (
    gaussian_component_1,
    gaussian_mixture_4d_logpdf,
)
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.main import run
from queens.utils import injector


def test_smc_bayes_temper_multivariate_gaussian_mixture(
    inputdir, tmp_path, _create_experimental_data
):
    """Test SMC with a multivariate Gaussian mixture (multimodal)."""
    template = Path(inputdir, "smc_bayes_temper_multivariate_gaussian_mixture.yml")
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "multivariate_gaussian_mixture_smc_bayes_temper_realiz.yml"
    injector.inject(dir_dict, template, input_file)

    # mock methods related to likelihood
    with patch.object(SequentialMonteCarloIterator, "eval_log_likelihood", target_density):
        with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
            run(input_file, tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [-0.4 -0.4 -0.4 -0.4]
    # posterior var: [0.1, 0.1, 0.1, 0.1]
    # however, we only have a very inaccurate approximation here:
    np.testing.assert_almost_equal(
        results['mean'], np.array([[0.23384, 0.21806, 0.24079, 0.24528]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['var'], np.array([[0.30894, 0.15192, 0.19782, 0.18781]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['cov'],
        np.array(
            [
                [
                    [0.30894, 0.21080, 0.24623, 0.23590],
                    [0.21080, 0.15192, 0.17009, 0.15951],
                    [0.24623, 0.17009, 0.19782, 0.18695],
                    [0.23590, 0.15951, 0.18695, 0.18781],
                ]
            ]
        ),
        decimal=5,
    )


def target_density(self, samples):  # pylint: disable=unused-argument
    """TODO_doc."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_mixture_4d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture(name="_create_experimental_data")
def fixture_create_experimental_data(tmp_path):
    """TODO_doc."""
    # generate 10 samples from the same gaussian
    samples = gaussian_component_1.draw(10)
    pdf = gaussian_mixture_4d_logpdf(samples)

    pdf = np.array(pdf)

    # write the data to a csv file in tmp_path
    data_dict = {'y_obs': pdf}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)