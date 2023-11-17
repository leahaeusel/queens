"""Job interface class."""

from queens.interfaces.interface import Interface


class JobInterface(Interface):
    """Class for mapping input variables to responses.

    Attributes:
        scheduler (Scheduler):      scheduler for the simulations
        driver (Driver):            driver for the simulations
    """

    def __init__(self, parameters, scheduler, driver):
        """Create JobInterface.

        Args:
            parameters (obj):           Parameters object
            scheduler (Scheduler):      scheduler for the simulations
            driver (Driver):            driver for the simulations
        """
        super().__init__(parameters)
        self.scheduler = scheduler
        self.driver = driver
        self.scheduler.copy_file(self.driver.simulation_input_template)

    def evaluate(self, samples):
        """Evaluate.

        Args:
            samples (np.array): Samples of simulation input variables

        Returns:
            output (dict): Output data
        """
        samples_list = self.create_samples_list(samples)
        output = self.scheduler.evaluate(samples_list, driver=self.driver)

        return output