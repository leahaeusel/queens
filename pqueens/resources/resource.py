"""
The resource module contains everything that is necessary to manage computing resources.

A computing ressource can be a single machine or a HPC cluster.
The resource can provide basic status information as well as workload capacity.
If the workload capacity allows it the computing resource accepts jobs and executes them.
"""
import sys

import numpy as np

from pqueens.schedulers.scheduler import Scheduler

# TODO refactor this method into a class method


def parse_resources_from_configuration(config):
    """ Parse the configuration dictionary

    Args:
        config (dict): Dictionary with problem description
    Returns:
        dict: Dictionary with resource objects keyed with resource name
    """

    if "resources" in config:
        resources = dict()
        for resource_name, _ in config["resources"].items():
            global_settings = config.get("global_settings")
            exp_name = global_settings.get("experiment_name")
            resources[resource_name] = resource_factory(resource_name, exp_name, config)
        return resources
    # no specified resources
    else:
        raise Exception("Resources are not properly specified")


def resource_factory(resource_name, exp_name, config):
    """ Create a resource object

    Args:
        resource_name (string): name of resource
        exp_name (string):      name of experiment to be run on resource
        config   (dict):        dictionary with problem description

    Returns:
        resource:  resource object constructed from the resource name,
                   exp_name, and config dict
    """

    # get resource options extract resource info from config
    resource_options = config["resources"][resource_name]
    max_concurrent = resource_options.get('max-concurrent', 1)
    max_finished_jobs = resource_options.get('max-finished-jobs', np.inf)

    scheduler_name = resource_options['scheduler']

    # create scheduler from config
    scheduler = Scheduler.from_config_create_scheduler(scheduler_name=scheduler_name, config=config)
    # Create/update singularity image in case of cluster job
    scheduler.pre_run()

    return Resource(resource_name, exp_name, scheduler, max_concurrent, max_finished_jobs)


def print_resources_status(resources, jobs):
    """ Print out whats going on on the resources
    Args:
        resources (dict):   Dictionary with used resources
        jobs (list):        List of jobs

    """
    sys.stdout.write('\nResources:      ')
    left_indent = 16
    indentation = ' ' * left_indent
    sys.stdout.write('NAME          PENDING    COMPLETE    FAILED\n')
    sys.stdout.write(indentation)
    sys.stdout.write('----          -------    --------    ------\n')
    total_pending = 0
    total_complete = 0
    total_failed = 0

    # for resource in resources:
    for _, resource in resources.items():
        pending = resource.num_pending(jobs)
        complete = resource.num_complete(jobs)
        failed = resource.num_failed(jobs)
        total_pending += pending
        total_complete += complete
        total_failed += failed
        sys.stdout.write(
            "%s%-12.12s  %-9d  %-10d  %-9d\n"
            % (indentation, resource.name, pending, complete, failed)
        )
    sys.stdout.write(
        "%s%-12.12s  %-9d  %-10d  %-9d\n"
        % (indentation, '*TOTAL*', total_pending, total_complete, total_failed)
    )
    sys.stdout.write('\n')


class Resource(object):
    """class which manages a computing resource

    Attributes:
        name (string):                The name of the resource
        exp_name (string):            The name of the experiment
        scheduler (scheduler object): The object which submits and polls jobs
        scheduler_class (class type): The class type of scheduler.  This is used
                                      just for printing

        max_concurrent (int):         The maximum number of jobs that can run
                                      concurrently on resource

        max_finished_jobs (int):      The maximum number of jobs that can be
                                      run to completion
    """

    def __init__(self, name, exp_name, scheduler, max_concurrent, max_finished_jobs):
        """
        Args:
            name (string):                The name of the resource
            exp_name (string):            The name of the experiment
            scheduler (scheduler object): The object which submits and polls jobs
            max_concurrent (int):         The maximum number of jobs that can run
                                          concurrently on resource

            max_finished_jobs (int):      The maximum number of jobs that can be
                                          run to completion
        """
        self.name = name
        self.scheduler = scheduler
        self.scheduler_class = type(scheduler).__name__  # stored just for printing
        self.max_concurrent = max_concurrent
        self.max_finished_jobs = max_finished_jobs
        self.exp_name = exp_name

        if len(self.exp_name) == 0:
            sys.stdout.write("Warning: resource %s has no tasks assigned " " to it" % self.name)

    def filter_my_jobs(self, jobs):
        """ Take a list of jobs and filter those that are on this resource

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs belonging to this resource

        """
        if jobs:
            return [job for job in jobs if job['resource'] == self.name]
            # filter(lambda job: job['resource']==self.name, jobs)
        return jobs

    def num_pending(self, jobs):
        """ Take a list of jobs and filter those that are either pending or new

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs that are either pending or new

        """
        jobs = self.filter_my_jobs(jobs)
        if jobs:
            pending_jobs = [job['status'] for job in jobs].count('pending')
            return pending_jobs
        return 0

    def num_failed(self, jobs):
        """ Take a list of jobs and filter those that have failed

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs that have failed

        """
        jobs = self.filter_my_jobs(jobs)
        if jobs:
            failed_jobs = [job['status'] for job in jobs].count('failed')
            # map(lambda x: x['status'] in ['failed'], jobs)
            return failed_jobs  # reduce(add, failed_jobs, 0)
        else:
            return 0

    def num_complete(self, jobs):
        """ Take a list of jobs and filter those that are complete

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs that either are complete

        """
        jobs = self.filter_my_jobs(jobs)
        if jobs:
            completed_jobs = [job['status'] for job in jobs].count('complete')
            # map(lambda x: x['status'] == 'complete', jobs)
            return completed_jobs  # reduce(add, completed_jobs, 0)
        else:
            return 0

    def accepting_jobs(self, jobs):
        """ Check if the resource currently is accepting new jobs

        Args:
            jobs (list): List with jobs

        Returns:
            bool: whether or not resource is accepting jobs

        """
        if self.num_pending(jobs) >= self.max_concurrent:
            return False

        if self.num_complete(jobs) >= self.max_finished_jobs:
            return False

        return True

    def print_status(self, jobs):
        """ Print number of pending ans completed jobs

        Args:
            jobs (list): List with jobs
        """
        sys.stdout.write(
            "%-12s: %5d pending %5d complete\n"
            % (self.name, self.num_pending(jobs), self.num_complete(jobs))
        )

    def is_job_alive(self, job):  # TODO this method does not seem to be called
        """ Query if a particular job is alive?

        Args:
            job (dict): jobs to query

        Returns:
            bool: whether or not job is alive

        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        return self.scheduler.alive(job['proc_id'])

    def attempt_dispatch(self, batch, job):
        """ Submit a new job using the scheduler of the resource

        Args:
            batch (string):         Batch number of job
            job (dict):             Job to submit

        Returns:
            int:       Process ID of job
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        process_id = self.scheduler.submit(job['id'], batch)
        if process_id is not None:
            if process_id != 0:
                sys.stdout.write(
                    'Submitted job %d with %s '
                    '(process id: %d).\n' % (job['id'], self.scheduler_class, process_id)
                )
            elif process_id == 0:
                sys.stdout.write(
                    'Checked job %d for restart and loaded results into database.\n\n' % job['id']
                )
        else:
            sys.stdout.write('Failed to submit job %d.\n' % job['id'])

        return process_id

    def dispatch_post_post_job(self, batch, job):
        """ Submit a new post-post job using the scheduler of the resource

        Args:
            batch (string):         Batch number of job
            job (dict):             Job to submit

        Returns:
            int:       Process ID of job
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        self.scheduler.submit_post_post(job['id'], batch)
        sys.stdout.write(
            'Submitted post-post job %d with %s \n' % (job['id'], self.scheduler_class)
        )
