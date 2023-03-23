"""Test NUTS Iterator."""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from pqueens import run
from pqueens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_logpdf import (
    gaussian_2d_logpdf,
)
from pqueens.utils import injector


def test_gaussian_nuts(inputdir, tmpdir, dummy_data):
    """Test case for nuts iterator."""
    template = os.path.join(inputdir, "nuts_gaussian.yml")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = os.path.join(tmpdir, "gaussian_nuts_realiz.yml")
    injector.inject(dir_dict, template, input_file)
    with patch.object(GaussianLikelihood, "evaluate_and_gradient", target_density):
        run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    assert results['mean'].mean(axis=0) == pytest.approx(
        np.array([-0.2868793496608573, 0.6474274597130008])
    )
    assert results['var'].mean(axis=0) == pytest.approx([0.08396277217936474, 0.10836256575521087])


def target_density(self, samples):
    """Patch likelihood."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_2d_logpdf(samples).flatten()

    cov = [[1.0, 0.5], [0.5, 1.0]]
    cov_inverse = np.linalg.inv(cov)
    gradient = -np.dot(cov_inverse, samples.T).T

    return (log_likelihood, gradient)


@pytest.fixture()
def dummy_data(tmpdir):
    """Generate 2 samples from the same gaussian."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': samples}
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)