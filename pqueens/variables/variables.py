"""
Variables module provides a data container for random variables and their realizations
"""
import numpy as np

from pqueens.utils import mcmc_utils


class Variables(object):
    """ Class for storing variables

    For now basically only a wrapper around a dictionary with two sub dictionaries.
    One for random variables and one for random fields.

    At this point random variables are simply a dictionary with three fields.
    The key is the name of the variable and then we have a 'type' field, a 'value'
    field and a 'active' field. The same basically applies to random fields,
    with the difference being that multiple values are stored.

    Attributes:
        variables (dict):  dictionary containing the data
    """

    def __init__(self, uncertain_parameters, values=None, active=None):
        """ Initialize variable object

        Args:
            uncertain_parameters (dict): description of all uncertain params
            values (list):               list with variable values
            active (list):               list with flag whether or not variable
                                         is active
        """
        self.variables = {}  # TODO Check if we need here: <uncertain_parameters> instead of <{}>
        i = 0
        for key, data in uncertain_parameters["random_variables"].items():
            # TODO Check if the following lines are necessary in other scenarios
            self.variables[key] = {}
            if data.get('size'):  # TODO workaround to make BMFMC work
                my_size = data['size']
                self.variables[key]['size'] = my_size
                self.variables[key]['value'] = values[i : i + my_size]
                self.variables[key]['type'] = data['type']
                self.variables[key]['distribution'] = mcmc_utils.create_proposal_distribution(data)
                self.variables[key]['active'] = active[i]
                i += my_size
            else:
                self.variables = uncertain_parameters
                self.variables['random_variables'][key].update({'active': True})

        if uncertain_parameters.get("random_fields") is not None:
            for key, data in uncertain_parameters["random_fields"].items():
                self.variables[key] = {}
                dim = data["dimension"]
                eval_locations_list = data.get("eval_locations", None)
                eval_locations = np.array(eval_locations_list).reshape(-1, dim)
                my_size = eval_locations.shape[0]
                self.variables[key]['size'] = my_size
                self.variables[key]['value'] = values[i : i + my_size]
                self.variables[key]['type'] = data['type']
                self.variables[key]['active'] = active[i]
                i += my_size

    @classmethod
    def from_uncertain_parameters_create(cls, uncertain_parameters):
        """ Create variables from uncertain parameter

        Args:
            uncertain_parameters (dict): Dictionary with uncertain parameters

        Returns:
            variables: Instance of variables object
        """
        values = []
        active = []
        for _, _ in uncertain_parameters["random_variables"].items():
            values.append(None)
            active.append(True)
        if uncertain_parameters.get("random_fields") is not None:
            for _, _ in uncertain_parameters["random_fields"].items():
                values.append(None)
                active.append(True)
        return cls(uncertain_parameters, values, active)

    @classmethod
    def from_data_vector_create(cls, uncertain_parameters, data_vector):
        """ Create variables from uncertain parameter

        Args:
            uncertain_parameters (dict): Dictionary with uncertain parameters
            data_vector (np.array):      Vector with variable values

        Returns:
            variables: Instance of variables object
        """

        values = data_vector.tolist()
        # TODO fix this
        active = [True] * len(values)

        return cls(uncertain_parameters, values, active)

    def get_active_variables(self):
        """ Get dictinary of all active variables

        Returns:
            dict: dictionary with active variables, name and value only
        """
        active_vars = {}
        for key, data in self.variables.items():
            if data['active'] is not True:
                continue
            # TODO store value entries as list or make sure that is
            # other wise compatiple
            # if len(data['value']) > 1:
            #    active_vars[key] = data['value'].tolist()
            # else:
            active_vars[key] = data['value']
        return active_vars

    def get_active_variables_vector(self):
        """ Get vector with values of all active variables

        Returns:
            np.array: vector with values of all active variables
        """
        active_var_vals = [data["value"] for _, data in self.variables.items() if data["active"]]
        return np.hstack(active_var_vals)

    def get_number_of_active_variables(self):
        """ Get number of currently active variables

        Returns:
            int: number of active variables
        """
        num_active_vars = 0
        for _, data in self.variables.items():
            if data['active'] is not True:
                continue
            num_active_vars += 1
        return num_active_vars

    def update_variables(self, new_variable_data):
        """ Update variable data

        Args:
            new_variable_data (dict): data to update the variables with
        """
        # TODO: add support for updating the distribution
        for key, data in self.variables.items():
            data['size'] = new_variable_data[key]['size']
            data['value'] = new_variable_data[key]['value']
            data['active'] = new_variable_data[key]['active']
            data['type'] = new_variable_data[key]['type']
            data['distribution'] = mcmc_utils.create_proposal_distribution(new_variable_data[key])

    def update_variables_from_vector(self, data_vector):
        """ Update variable values from vector

        Args:
            data_vector (np.array): Vector with new data for active variables
        """
        i = 0
        for key, _ in self.variables.items():
            my_size = self.variables[key]['size']
            self.variables[key]['value'] = np.squeeze(
                data_vector[i : i + my_size]
            )  # TODO here might be is a problem!
            i += my_size
        if i != len(data_vector):
            raise IndexError('The passed vector is to long!')
