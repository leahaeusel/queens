"""TODO_doc."""

import pickle

import numpy as np
import pytest

from queens.main import run


@pytest.mark.max_time_for_test(30)
def test_branin_gpflow_heteroskedastic(inputdir, tmp_path, expected_mean, expected_var):
    """Test case for GPflow based heteroskedastic model."""
    run(inputdir / 'gp_heteroskedastic_surrogate_branin.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"], expected_mean, decimal=2
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """TODO_doc."""
    mean = np.array(
        [
            [5.12899758],
            [4.07705793],
            [10.22698405],
            [2.55114341],
            [4.56170606],
            [2.45214928],
            [2.5610184],
            [3.32166064],
            [7.84211279],
            [6.96924365],
        ]
    )
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """TODO_doc."""
    var = np.array(
        [
            [1057.67077678],
            [4803.47277567],
            [1298.02323316],
            [1216.18467643],
            [456.57239924],
            [13143.69339927],
            [8244.44914723],
            [21364.37227331],
            [877.20094392],
            [207.58530761],
        ]
    )
    return var