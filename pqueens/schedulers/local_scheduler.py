import os
import sys
from .scheduler import Scheduler


class LocalScheduler(Scheduler):
    """
    Scheduler for QUEENS jobs on the localhost.

    Args:
        base_settings (dict): Configurations that are set in the base class and are partly used
                              in this class
        scheduler_name (str): Name of the scheduler as specified in input file

    Attributes:
        name (str): Name of the scheduler as specified in input file
    """

    def __init__(self, base_settings, scheduler_name):
        self.name = scheduler_name
        super(LocalScheduler, self).__init__(base_settings)

    @classmethod
    def from_config_create_scheduler(
        cls, config, base_settings, scheduler_name=None
    ):  # TODO scheduler name
        # is depreciated
        """ Create scheduler from config dictionary

        Args:
            scheduler_name (str):   (Optional) Name of scheduler
            config (dict):          Dictionary containing problem description of input file
            base_settings (dict): Configurations that are set in the base class and are partly
                                  reused to construct this class

        Returns:
            scheduler_obj (obj): Instance of LocalScheduler

        """

        # pre assemble some strings as base_settings
        base_settings['output'] = None
        base_settings['tasks_info'] = None
        base_settings['walltime_info'] = None
        base_settings['job_flag'] = None
        base_settings['scheduler_start'] = None
        base_settings['command_line_opt'] = None
        base_settings['scheduler_options'] = None

        return cls(base_settings, scheduler_name)

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def alive(self, process_id):  # TODO: seems not to be used
        """
        Check whether or not job is still running

        Args:
            process_id (int): id of process associated to job

        Returns:
            bool: indicator if job is still alive
        """

        alive = False
        command_list = ['ps h -p', str(process_id)]
        command_string = ' '.join(command_list)
        stdout, _, p = super().run_subprocess(command_string)

        if stdout:
            sys.stderr.write("Job %d waiting in queue.\n" % (process_id))
            alive = True
        else:
            sys.stderr.write("Job %d is held or suspended.\n" % (process_id))
            alive = False

        if not alive:
            try:
                # try to kill the job.
                os.kill(process_id, 0)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except ValueError:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True

    def get_process_id_from_output(self):
        """
        Not necessary for local scheduler but mandatory for parent class initialization

        Returns:
            None

        """
        pass
