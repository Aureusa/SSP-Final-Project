from utils import get_data
import jax.numpy as jnp

from plotting.plotting_utils import Plotter
from estimator.estimator import MLEGeneralJAX
from model.model_constructor import ModelConstructor
from markov_chain_monte_carlo.mc_simulator import MCSimulator
from utils import print_mc_results


freq, signal, signal_noise = get_data()
nu = jnp.array(freq.tolist())
s_n = jnp.array(signal_noise.tolist())
DATA = tuple((nu, s_n))

INITIAL_GUESS = [6.0, 2.0, 1.0]

CHECKPOINTS = list(range(1000, 10001, 1000))


def ex_1():
    """
    Exercise 1: Plots the model function, including the signal and noisy signal.

    Utilizes a `Plotter` object to visualize the frequency-domain signal data.
    """
    plotter = Plotter()

    plotter.plot_model_function(freq, signal, signal_noise)


def ex_2():
    """
    Exercise 2: Constructs and evaluates the Maximum Likelihood Estimator (MLE) model.

    This exercise demonstrates how to:
    - Create an MLE estimator with a generative model and log-likelihood function
    - Evaluate the ML estimate with initial guesses
    - Plot gradient norms, parameter estimates, and residuals

    Outputs the estimated parameters and visualizations of the optimization process.
    """
    model_constructor = ModelConstructor()

    log_likelihood = model_constructor.get_log_likelihood()
    generative_model = model_constructor.get_generative_model()

    nuisance_params = ["v", "s_n", "sigma"]

    mle_estimator = MLEGeneralJAX(
        model=generative_model,
        log_likelihood=log_likelihood,
        nuisance_params=nuisance_params,
        data=DATA,
        gamma=0.1,
    )

    guess = mle_estimator.evaluate_the_ML_estimate(
        guess=INITIAL_GUESS, data_logging=True, treshhold=1e-3
    )

    mle_estimator.plot_gradient_norms()
    mle_estimator.plot_estimates()
    mle_estimator.plot_residuals(guess)

    print(guess)


def ex_3(
    perform_sim: bool = False, num_cpus: int = 8, num_realizations: int = 5
):
    """
    Exercise 3: Runs a Monte Carlo simulation and visualizes the results.

    This exercise demonstrates how to:
    - Perform a Monte Carlo simulation of the model
    - Visualize realizations and parameter estimates
    - Plot Newton-Raphson optimization estimates
    - Display summary statistics of parameter estimates

    :param perform_sim: Whether to perform the Monte Carlo simulation, defaults to False
    :type perform_sim: bool, optional
    :param num_cpus: Number of CPU cores to use for the simulation, defaults to 8
    :type num_cpus: int, optional
    :param num_realizations: Number of realizations to visualize, defaults to 5
    :type num_realizations: int, optional
    """
    simulator = MCSimulator(freq, signal)

    if perform_sim:
        simulator.perform_simulation(num_cpus)

    simulator.plot_mc_realizations(num_realizations)

    simulator.plot_estimates_duiring_nr_optimization()

    (mean_A, var_A, mean_v0, var_v0, mean_alpha, var_alpha) = (
        simulator.plot_estimates_pdf()
    )

    print_mc_results(mean_A, var_A, mean_v0, var_v0, mean_alpha, var_alpha)


if __name__ == "__main__":
    ...
