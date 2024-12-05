"""Driver to recycle the model evaluations from a previous QUEENS run."""

import logging
import os

from queens.drivers.driver import Driver
from queens.drivers.jobscript_driver import JobscriptDriver
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata, get_metadata_from_job_dir

_logger = logging.getLogger(__name__)


class RecycleDriver(JobscriptDriver):
    """Driver to recycle the model evaluations from a previous QUEENS run.

    Attributes:
        input_template (Path): Read in simulation input template as string
        data_processor (obj): Instance of data processor class
        gradient_data_processor (obj): Instance of data processor class for gradient data
        jobscript_template (str): Read in jobscript template as string
        jobscript_options (dict): Dictionary containing jobscript options
        jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        data_processor=None,
        gradient_data_processor=None,
    ):
        """Initialize JobscriptDriver object.

        Args:
            parameters (Parameters): Parameters object
            data_processor (obj, opt): Instance of data processor class
            gradient_data_processor (obj, opt): Instance of data processor class for gradient data
        """
        # pylint: disable=non-parent-init-called, super-init-not-called

        Driver.__init__(self, parameters=parameters)

        self.data_processor = data_processor
        self.gradient_data_processor = gradient_data_processor

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample (dict): Sample for which to run the driver
            job_id (int): Job ID
            num_procs (int): Number of processors
            experiment_dir (Path): Path to QUEENS experiment directory
            experiment_name (str): Name of QUEENS experiment

        Returns:
            Result and potentially the gradient
        """
        job_dir = experiment_dir / str(job_id)
        output_dir = job_dir / "output"
        inputs = self.parameters.sample_as_dict(sample)

        # Make sure input parameters are the same
        original_metadata = get_metadata_from_job_dir(job_dir)

        if not original_metadata["inputs"] == inputs:
            raise ValueError(
                f"Input parameters of the original QUEENS run and the Recycle Driver run must be "
                f"equal for job ID {job_id}"
            )

        new_metadata_filename = self._new_metadata_filename(job_dir)
        metadata = SimulationMetadata(
            job_id=job_id, inputs=inputs, job_dir=job_dir, metadata_filename=new_metadata_filename
        )

        with metadata.time_code("data_processing"):
            results = self._get_results(output_dir)
            metadata.outputs = results

        return results

    def _new_metadata_filename(self, job_dir):
        """Find the next metadata file name.

        This is to avoid overwriting of already existing metadata files

        Returns:
            str: New metadata filename
        """
        recycle_run = 1
        metadata_filename = self.metadata_filename(recycle_run)

        while os.path.isfile(job_dir / (metadata_filename + ".yaml")):
            recycle_run += 1
            metadata_filename = self.metadata_filename(recycle_run)

        return metadata_filename

    @staticmethod
    def metadata_filename(recycle_run):
        """Metadata filename for a specific run of the recycle driver.

        Args:
            recycle_run (int): Counter of the number of recycle runs

        Returns:
            str: metadata filename of the recycle run
        """
        return f"metadata_recycle_run_{recycle_run}"
