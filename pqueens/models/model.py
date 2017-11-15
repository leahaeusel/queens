import abc
from pqueens.variables.variables import Variables
from copy import deepcopy

class Model(metaclass=abc.ABCMeta):
    """ Base class of model hierarchy

        The model hierarchy contains a set of variables, an interface,
        and a set of responses. An iterator operates on the model to map
        the variables into responses using the interface.

        As with the Iterator hierarchy, the purpose of the this base class is
        twofold. One, it defines a unified interface for all derived classes.
        Two, it acts as a factory for the instanciation of model objects.

    Attributes:
        interface (interface):          Interface to simulations/functions
        uncertain_parameters (dict):    Dictionary with description of uncertain
                                        pararameters
        variables (list):               Set of model variables where model is evaluated
        responses (list):               Set of responses corresponding to variables
    """

    def __init__(self, name, interface, uncertain_parameters):
        """ Init model onject

        Args:
            name (string):                  Name of model
            interface (interface):          Interface to simulations/functions
            uncertain_parameters (dict):    Dictionary with description of uncertain
                                            pararameters
        """
        self.name = name
        self.interface = interface
        self.uncertain_parameters = uncertain_parameters
        self.variables = [Variables.from_uncertain_parameters_create(uncertain_parameters)]
        self.response = [None]

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """ Create model from problem description

        Args:
            model_name (string):    Name of model
            config  (dict):         Dictionary with problem description

        Returns:
            model: Instance of model class

        """
        model_options = config[model_name]
        model_class = model_dict[model_options["type"]]
        return model_class.from_config_create_model(model_name, config)


    @abc.abstractmethod
    def evaluate(self):
        """ Evaluate model with current set of variables """
        pass

    def get_parameter_distribution_info(self):
        """ Get probability distributions for all uncertain parameters

        Returns:
            list: List with probability distribution information

        """
        distribution_info = []
        for _, value in self.uncertain_parameters.items():
            distribution_info.append(value)

        return distribution_info

    def get_parameter_names(self):
        """ Get names of all uncertain parameters

        Returns:
            list: List with names of uncertain parameters

        """
        parameter_names = []
        for key, _ in self.uncertain_parameters.items():
            parameter_names.append(key)

        return parameter_names

    def get_parameter(self):
        """ Get complete parameter dictionary

        Return:
            dict: Dictionary with all pararameters

        """
        return self.uncertain_parameters

    def update_model_from_sample(self, data_vector):
        """ Update model variables

        Args:
            data_vector (np.array): Vector with variable values

        """
        if len(self.variables) != 1:
            self.variables = deepcopy([self.variables[0]])

        self.variables[0].update_variables_from_vector(data_vector)

    def update_model_from_sample_batch(self, data):
        """ Update model variables

        Args:
            data (np.array): 2d array with variable values

        """
        temp = self.variables[0]
        temp = deepcopy(self.variables[0])
        self.variables = []
        for i in range(data.shape[0]):
            data_vector = data[i, :]
            temp.update_variables_from_vector(data_vector)
            new_var = deepcopy(temp)
            self.variables.append(new_var)


from .simulation_model import SimulationModel
model_dict = {'simulation_model': SimulationModel}