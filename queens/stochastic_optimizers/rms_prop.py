"""RMSprop optimizer."""

import logging

import numpy as np

from queens.stochastic_optimizers.stochastic_optimizer import StochasticOptimizer
from queens.utils.iterative_averaging_utils import ExponentialAveraging

_logger = logging.getLogger(__name__)


class RMSprop(StochasticOptimizer):
    r"""RMSprop stochastic optimizer [1].

    References:
        [1] Tieleman and Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of
            its recent magnitude". Coursera. 2012.

    Attributes:
        beta (float):  :math:`\beta` parameter as described in [1].
        v (ExponentialAveragingObject): Exponential average of the gradient momentum.
        eps (float): Nugget term to avoid a division by values close to zero.
    """

    _name = "RMSprop Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        rel_l1_change_threshold=None,
        rel_l2_change_threshold=None,
        clip_by_l2_norm_threshold=1e6,
        clip_by_value_threshold=1e6,
        max_iteration=1e6,
        beta=0.999,
        eps=1e-8,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            rel_l1_change_threshold (float): If the L1 relative change in parameters falls below
                                             this value, this criteria catches.
            rel_l2_change_threshold (float): If the L2 relative change in parameters falls below
                                            this value, this criteria catches.
            clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm
            clip_by_value_threshold (float): Threshold to clip the gradient components
            max_iteration (int): Maximum number of iterations
            beta (float): :math:`beta` parameter as described in [1]
            eps (float): Nugget term to avoid a division by values close to zero
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
            rel_l1_change_threshold=rel_l1_change_threshold,
            rel_l2_change_threshold=rel_l2_change_threshold,
            clip_by_l2_norm_threshold=clip_by_l2_norm_threshold,
            clip_by_value_threshold=clip_by_value_threshold,
            max_iteration=max_iteration,
        )
        self.beta = beta
        self.v = ExponentialAveraging(coefficient=beta)
        self.eps = eps

    def scheme_specific_gradient(self, gradient):
        """Rmsprop gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): RMSprop gradient
        """
        if self.iteration == 0:
            self.v.current_average = np.zeros(gradient.shape)

        v_hat = self.v.update_average(gradient**2)
        v_hat /= 1 - self.beta ** (self.iteration + 1)
        gradient = gradient / (v_hat**0.5 + self.eps)
        return gradient