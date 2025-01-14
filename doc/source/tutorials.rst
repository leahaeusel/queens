.. _tutorials:

Tutorials and examples
==================================

Work in progress, but here a simple example from our `README.md <https://github.com/queens-py/queens/blob/main/README.md>`_:

.. code-block:: python

   from queens.distributions import BetaDistribution, NormalDistribution, UniformDistribution
   from queens.drivers import FunctionDriver
   from queens.global_settings import GlobalSettings
   from queens.iterators import MonteCarloIterator
   from queens.main import run_iterator
   from queens.models import SimulationModel
   from queens.parameters import Parameters
   from queens.schedulers import LocalScheduler

   if __name__ == "__main__":
       # Set up the global settings
       global_settings = GlobalSettings(experiment_name="monte_carlo_uq", output_dir=".")

       # Set up the uncertain parameters
       x1 = UniformDistribution(lower_bound=-3.14, upper_bound=3.14)
       x2 = NormalDistribution(mean=0.0, covariance=1.0)
       x3 = BetaDistribution(lower_bound=-3.14, upper_bound=3.14, a=2.0, b=5.0)
       parameters = Parameters(x1=x1, x2=x2, x3=x3)

       # Set up the model
       driver = FunctionDriver(parameters=parameters, function="ishigami90")
       scheduler = LocalScheduler(
           experiment_name=global_settings.experiment_name, num_jobs=2, num_procs=4
       )
       model = SimulationModel(scheduler=scheduler, driver=driver)

       # Set up the algorithm
       iterator = MonteCarloIterator(
           model=model,
           parameters=parameters,
           global_settings=global_settings,
           seed=42,
           num_samples=1000,
           result_description={"write_results": True, "plot_results": True},
       )

       # Start QUEENS run
       run_iterator(iterator, global_settings=global_settings)
