"""Driver to recycle the model evaluations from a previous QUEENS run."""

import logging

from queens.drivers.jobscript_driver import JobscriptDriver
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class RecycleDriver(JobscriptDriver):
    """Driver to recycle the model evaluations from a previous QUEENS run.

    Attributes:
        input_template (Path): read in simulation input template as string
        data_processor (obj): instance of data processor class
        gradient_data_processor (obj): instance of data processor class for gradient data
        jobscript_template (str): read in jobscript template as string
        jobscript_options (dict): Dictionary containing jobscript options
        jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
    """

    @log_init_args
    def __init__(
        self,
        data_processor=None,
        gradient_data_processor=None,
    ):
        """Initialize JobscriptDriver object.

        Args:
            input_template (str, Path): path to simulation input template
            jobscript_template (str, Path): path to jobscript template or read in jobscript template
            executable (str, Path): path to main executable of respective software
            files_to_copy (list, opt): files or directories to copy to experiment_dir
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
            extra_options (dict): Extra options to inject into jobscript template
        """
        super().__init__(
            input_template=None,
            jobscript_template="",
            executable=None,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
        )

    def run(self, sample_dict, num_procs, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        job_id = sample_dict.pop("job_id")
        job_dir = experiment_dir / str(job_id)
        output_dir = job_dir / "output"

        results = self._get_results(output_dir)

        return results
