"""Test chaospy wrapper."""
import os
import pickle
from pathlib import Path

import pytest

from pqueens import run


@pytest.mark.integration_tests
def test_polynomial_chaos_pseudo_spectral_borehole(inputdir, tmpdir):
    """Test case for the pc iterator using a pseudo spectral approach."""
    run(
        Path(os.path.join(inputdir, 'polynomial_chaos_pseudo_spectral_borehole.json')), Path(tmpdir)
    )

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(61.78966587)
    assert results["covariance"] == pytest.approx([1312.23414971])


@pytest.mark.integration_tests
def test_polynomial_chaos_collocation_borehole(inputdir, tmpdir):
    """Test for the pc iterator using a collocation approach."""
    run(Path(os.path.join(inputdir, 'polynomial_chaos_collocation_borehole.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05018243)
    assert results["covariance"] == pytest.approx([1273.81372103])