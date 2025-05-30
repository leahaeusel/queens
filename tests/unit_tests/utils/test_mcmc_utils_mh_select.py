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
"""Test-module for *mh_select* function of *mcmc_utils module*.

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest

from queens.utils.mcmc import mh_select


@pytest.fixture(name="acceptance_probability", scope="module")
def fixture_acceptance_probability():
    """Possible acceptance probability."""
    return 0.3


@pytest.fixture(name="num_chains", scope="module")
def fixture_num_chains():
    """Number of parallel chains."""
    return 2


@pytest.fixture(name="log_acceptance_probability", scope="module")
def fixture_log_acceptance_probability(acceptance_probability, num_chains):
    """Possible natural logarithm of acceptance probability."""
    acceptance_probability = np.array([[acceptance_probability]] * num_chains)
    log_acceptance_probability = np.log(acceptance_probability)
    return log_acceptance_probability


@pytest.fixture(name="current_sample", scope="module")
def fixture_current_sample(num_chains):
    """A potential current sample of an MCMC chain."""
    return np.array([[3.0, 4.0]] * num_chains)


@pytest.fixture(name="proposed_sample", scope="module")
def fixture_proposed_sample(current_sample):
    """A potentially proposed sample of an MCMC algorithm."""
    return 2.0 * current_sample


def test_mh_select(
    acceptance_probability, log_acceptance_probability, current_sample, proposed_sample, mocker
):
    """Test Metropolis-Hastings selection process.

    Test rejection and acceptance of proposal based on given acceptance
    probability.
    """
    mocker.patch(
        "numpy.random.uniform",
        return_value=np.array([[acceptance_probability * 2], [acceptance_probability * 0.5]]),
    )

    selected_sample, accepted = mh_select(
        log_acceptance_probability, current_sample, proposed_sample
    )

    assert np.allclose(selected_sample[0], current_sample[0])
    assert not accepted[0]
    assert np.allclose(selected_sample[1], proposed_sample[1])
    assert accepted[1]


def test_mh_select_accept_prob_1(current_sample, proposed_sample, num_chains):
    r"""Test MH acceptance when acceptance probability is >= 1."""
    acceptance_probability = np.ones((num_chains, 1))
    log_acceptance_probability = np.log(acceptance_probability)
    selected_sample, accepted = mh_select(
        log_acceptance_probability, current_sample, proposed_sample
    )

    assert np.allclose(selected_sample, proposed_sample)
    assert np.all(accepted)


def test_mh_select_accept_prob_0(current_sample, proposed_sample, num_chains):
    """Test rejection of proposal based on acceptance probability = 0.0."""
    acceptance_probability = np.zeros((num_chains, 1))
    log_acceptance_probability = np.log(acceptance_probability)
    selected_sample, accepted = mh_select(
        log_acceptance_probability, current_sample, proposed_sample
    )

    assert np.allclose(selected_sample, current_sample)
    assert not np.all(accepted)
