"""Test dirichlet to dat."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


def test_write_random_dirichlet_to_dat(
    inputdir, tmp_path, third_party_inputs, baci_link_paths, expected_result
):
    """Dirichlet to dat file."""
    # generate json input file from template
    third_party_input_file = third_party_inputs / "baci_input_files/invaaa_ee_fields_template.dat"

    dat_file_preprocessed = tmp_path / "invaaa_ee_fields_template_preprocessed.dat"
    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'baci_input': third_party_input_file,
        'baci_input_preprocessed': dat_file_preprocessed,
        'post_drt_monitor': post_drt_monitor,
        'baci_release': baci_release,
    }
    template = inputdir / "baci_write_random_field_to_dat_template_dask.yml"
    input_file = tmp_path / "baci_write_random_field_to_dat.yml"
    injector.inject(dir_dict, template, input_file)

    # get json file as config dictionary
    run(Path(input_file), Path(tmp_path))

    # run a MC simulation with random input for now

    # Check if we got the expected results
    experiment_name = "baci_write_random_field_to_dat"
    result_file_name = experiment_name + ".pickle"

    result_file = tmp_path / result_file_name
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # assert statements
    np.testing.assert_array_almost_equal(
        results['raw_output_data']['mean'], expected_result, decimal=3
    )


@pytest.fixture()
def expected_result():
    """Expected result fixture."""
    result = np.array([[-0.04793531], [-0.04565255], [-0.04865387]]).reshape(3, 1)
    return result