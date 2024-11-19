"""Unit tests for the recycle driver."""

import numpy as np
import pytest

from queens.data_processor.data_processor_numpy import DataProcessorNumpy
from queens.distributions import UniformDistribution
from queens.drivers.jobscript_driver import JobscriptDriver
from queens.drivers.recycle_driver import RecycleDriver
from queens.parameters import Parameters
from queens.utils.metadata import get_metadata_from_job_dir


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Dummy parameters."""
    parameter_1 = UniformDistribution(lower_bound=0.0, upper_bound=2.0)
    parameter_2 = UniformDistribution(lower_bound=1.0, upper_bound=4.0)
    parameters = Parameters(parameter_1=parameter_1, parameter_2=parameter_2)

    return parameters


@pytest.fixture(name="sample")
def fixture_sample(parameters):
    """Dummy sample from the parameters."""
    sample_dict = parameters.sample_as_dict(np.array([1, 2]))
    sample = np.array(list(sample_dict.values()))
    return sample


@pytest.fixture(name="different_sample")
def fixture_different_sample(parameters):
    """Different dummy sample from the parameters."""
    sample_dict = parameters.sample_as_dict(np.array([0.5, 3.0]))
    sample = np.array(list(sample_dict.values()))
    return sample


@pytest.fixture(name="num_procs")
def fixture_num_procs():
    """Number of processors for the drivers."""
    return 1


@pytest.fixture(name="job_id")
def fixture_job_id():
    """Dummy job ID."""
    return 3


@pytest.fixture(name="experiment_name")
def fixture_experiment_name():
    """Dummy experiment name."""
    return "test_recycle_driver"


@pytest.fixture(name="dummy_data")
def fixture_dummy_data():
    """Dummy data."""
    data = np.array([[1.1, 2.1], [3.5, 4.5]])
    return data


@pytest.fixture(name="dummy_data_file")
def fixture_dummy_data_file():
    """Name of the dummy data file."""
    return "dummy_data.npy"


@pytest.fixture(name="_write_dummy_job_output")
def fixture_write_dummy_job_output(
    parameters, sample, job_id, num_procs, tmp_path, experiment_name, dummy_data, dummy_data_file
):
    """Conduct a dummy run with a jobscript driver.

    This is to create the associated files in the job directory, which
    are necessary for a recycle driver run. Since this dummy run does
    not produce any data in the output directory, dummy output data in
    the form of an npy file is added manually.
    """
    jobscript_driver = JobscriptDriver(
        parameters=parameters,
        jobscript_template="",
        executable="",
        input_templates={},
    )
    jobscript_driver.run(sample, job_id, num_procs, tmp_path, experiment_name)

    # Write dummy output data to output directory of this driver run
    job_dir = tmp_path / str(job_id)
    with open(job_dir / "output" / dummy_data_file, "wb") as f:
        np.save(f, dummy_data)


def test_recycle_driver_run_same_input_parameters(
    dummy_data_file,
    parameters,
    sample,
    job_id,
    num_procs,
    tmp_path,
    experiment_name,
    dummy_data,
    _write_dummy_job_output,
):
    """Test if the recycle driver runs correctly.

    This test uses the same input sample for the recycle driver run as
    for the previous jobscript driver run.
    """
    data_processor = DataProcessorNumpy(dummy_data_file, file_options_dict={})
    driver = RecycleDriver(
        parameters=parameters,
        data_processor=data_processor,
    )

    results = driver.run(sample, job_id, num_procs, tmp_path, experiment_name)

    job_dir = tmp_path / str(job_id)
    original_metadata = get_metadata_from_job_dir(job_dir)
    recycle_metadata = get_metadata_from_job_dir(
        job_dir, RecycleDriver.metadata_filename(recycle_run=1)
    )

    np.testing.assert_equal(results[0], dummy_data)
    np.testing.assert_equal(results, recycle_metadata["outputs"])
    np.testing.assert_equal(original_metadata["inputs"], recycle_metadata["inputs"])


def test_recycle_driver_run_different_input_parameters(
    dummy_data_file,
    parameters,
    different_sample,
    job_id,
    num_procs,
    tmp_path,
    experiment_name,
    _write_dummy_job_output,
):
    """Test if the recycle driver raises an error.

    This test uses a different input sample for the recycle driver run
    than for the previous jobscript driver run.
    """
    data_processor = DataProcessorNumpy(dummy_data_file, file_options_dict={})
    driver = RecycleDriver(
        parameters=parameters,
        data_processor=data_processor,
    )

    with pytest.raises(ValueError):
        driver.run(different_sample, job_id, num_procs, tmp_path, experiment_name)
