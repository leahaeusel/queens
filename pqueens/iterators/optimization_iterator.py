"""
Deterministic optimization toolbox

based on the scipy.optimize optimization toolbox [1]

References:
    [1]: https://docs.scipy.org/doc/scipy/reference/optimize.html

@author: Sebastian Brandstaeter
"""
import time

import numpy as np
import scipy.optimize

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.fd_jacobian import compute_step_with_bounds
from pqueens.utils.fd_jacobian import get_positions
from pqueens.utils.fd_jacobian import fd_jacobian
from pqueens.utils.process_outputs import write_results


class OptimizationIterator(Iterator):
    """
    Iterator for deteriministic optimization problems

    Attributes:
        initial_guess (np.array): initial guess, i.e. start point of
                                  optimization
        max_func_evals (int) : maximal number of function evaluations
        result_description (dict):  Description of desired
                                    post-processing
    """

    def __init__(self,
                 algorithm,
                 bounds,
                 constraints,
                 global_settings,
                 initial_guess,
                 max_func_evals,
                 model,
                 result_description,
                 ):
        super().__init__(model, global_settings)

        self.algorithm = algorithm
        self.bounds = bounds
        self.cons = constraints
        self.initial_guess = initial_guess
        self.max_func_evals = max_func_evals
        self.result_description = result_description

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None,
                                    model=None):
        """
        Create Optimization iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: OptimizationIterator object
        """

        print("Optimization Iterator for experiment: {0}"
              .format(config.get('global_settings').get('experiment_name')))
        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = Model.from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        initial_guess = np.array(method_options['initial_guess'])

        bounds = method_options.get("bounds", None)

        if bounds is None:
            bounds = [(-np.inf, np.inf)] * initial_guess.shape[0]

        constraints_dict = method_options.get('constraints', None)

        constraints = None
        if constraints_dict:
            constraints = list()
            for _,value in constraints_dict.items():
                # evaluate string of lambda function into real lambda function
                value['fun'] = eval(value['fun'])
                constraints.append(value)

        max_func_evals = method_options.get('max_func_evals', None)
        algorithm = method_options.get('algorithm', 'L-BFGS-B')
        algorithm = algorithm.upper()

        # initialize objective function
        return cls(algorithm=algorithm,
                   bounds=bounds,
                   constraints=constraints,
                   global_settings=global_settings,
                   max_func_evals=max_func_evals,
                   model=model,
                   result_description=result_description,
                   initial_guess=initial_guess)

    def eval_model(self):
        """ Evaluate model at current point. """

        result_dict = self.model.evaluate()
        return result_dict

    def eval_cost_function(self, x0):
        """
        Evaluate cost function at x0.
        """

        x_batch, _ = get_positions(x0, method='2-point', rel_step=None, bounds=self.bounds)

        self.model.update_model_from_sample_batch(x_batch)
        f_batch = np.squeeze(self.eval_model()['mean'])

        f0 = f_batch[0]

        return f0

    def eval_jacobian(self, x0, method='2-point', rel_step=None):
        """
        Evaluate Jacobian of cost function at x0.
        """

        positions, delta_positions = get_positions(x0, method=method, rel_step=rel_step, bounds=self.bounds)
        _, use_one_sided = compute_step_with_bounds(x0, method=method, rel_step=rel_step, bounds=self.bounds)

        response = self.model.get_precalculated_response_for_sample_batch(positions)
        f = response['mean']

        f0 = f[0] # first entry corresponds to f(x0)
        f_perturbed = np.delete(f, 0, 0) # delete the first entry

        J = fd_jacobian(f0, f_perturbed, delta_positions, use_one_sided, method=method)
        return J

    def initialize_run(self):
        """ Get initial guess. """

        print("Initialize Optimization run.")

    def core_run(self):
        """
        Core run of Optimization iterator
        """

        print('Welcome to Optimization core run.')
        start = time.time()
        # nonlinear least squares with bounds using Jacobian
        if self.algorithm == 'LSQ':
            self.solution = scipy.optimize.least_squares(self.eval_cost_function,
                                                         self.initial_guess,
                                                         jac=self.eval_jacobian,
                                                         bounds=self.bounds,
                                                         max_nfev=self.max_func_evals,
                                                         verbose=1)
        # minimization with bounds using Jacobian
        elif self.algorithm in {'L-BFGS-B', 'TNC'}:
            self.solution = scipy.optimize.minimize(self.eval_cost_function,
                                                    self.initial_guess,
                                                    method=self.algorithm,
                                                    jac=self.eval_jacobian,
                                                    bounds=self.bounds,
                                                    options={'maxiter' : int(1e4),
                                                             'disp' : True})
        # Constrained Optimimization BY Linear Approximation:
        # minimization with constraints without Jacobian
        elif self.algorithm in {'COBYLA'}:
            self.solution = scipy.optimize.minimize(self.eval_cost_function,
                                                    self.initial_guess,
                                                    method=self.algorithm,
                                                    constraints=self.cons,
                                                    options={'disp' : True})
        # Sequential Least SQuares Programming:
        # minimization with bounds and constraints using Jacobian
        elif self.algorithm in {'SLSQP'}:
            self.solution = scipy.optimize.minimize(self.eval_cost_function,
                                                    self.initial_guess,
                                                    method=self.algorithm,
                                                    jac=self.eval_jacobian,
                                                    bounds=self.bounds,
                                                    constraints=self.cons,
                                                    options={'disp' : True})
        # minimization (unconstrained, unbounded) without Jacobian
        elif self.algorithm in {'NELDER-MEAD', 'POWELL'}:
            self.solution = scipy.optimize.minimize(self.eval_cost_function,
                                                    self.initial_guess,
                                                    method=self.algorithm,
                                                    options={'disp' : True})
        # minimization (unconstrained, unbounded) using Jacobian
        elif self.algorithm in {'CG', 'BFGS'}:
            self.solution = scipy.optimize.minimize(self.eval_cost_function,
                                                    self.initial_guess,
                                                    method=self.algorithm,
                                                    jac=self.eval_jacobian,
                                                    options={'disp' : True})
        end = time.time()
        print(f"Optimization took {end-start} seconds.")

    def post_run(self):
        """ Analyze the resulting optimum. """

        print(f"The optimum:\n\t{self.solution.x}")
        if self.algorithm == 'LSQ':
            print(f"Optimality:\n\t{self.solution.optimality}")
            print(f"Cost:\n\t{self.solution.cost}")

        if self.result_description:
            if self.result_description["write_results"]:
                write_results(self.solution,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])

