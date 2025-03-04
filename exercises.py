from utils import get_data
import jax.numpy as jnp
import numpy as np

from bayesian_persuasion.bayes import BayesianAnalysis
from plotting.plotting_utils import Plotter
from estimator.estimator import MLEGeneralJAX
from model.model_constructor import ModelConstructor
from markov_chain_monte_carlo.mc_simulator import MCSimulator
from markov_chain_monte_carlo.mcmc import MCMC
from utils import print_mc_results
from estimator.crlb_estimator import CRLBEstimator


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
    mle_estimator.plot_fitted_model()
    mle_estimator.plot_residuals(guess)

    print(guess)


def ex_3(
    perform_sim: bool = False,
    num_cpus: int = 8,
    num_realizations: int = 5,
    folder: str = "results1_last",
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

    simulator.plot_estimates_duiring_nr_optimization(folder)

    simulator.plot_estimates_pdf(folder)

    mean_A, mean_v0, mean_alpha = simulator.get_means(folder)

    var_A, var_v0, var_alpha = simulator.get_variances(folder)

    print_mc_results(mean_A, var_A, mean_v0, var_v0, mean_alpha, var_alpha)


def ex_4(folder: str = "results1_last"):
    """
    Example 4: Computes the Fisher Information Matrix
    and the Cramér-Rao Lower Bound (CRLB) for a given set of parameters and data.
    Compares the CRLB with the variances obtained
    from simulations.
    """
    estimator = CRLBEstimator()

    simulator = MCSimulator(freq, signal)

    mean_A, mean_v0, mean_alpha = simulator.get_means(folder)

    parameters = tuple((mean_A, mean_v0, mean_alpha))

    data = np.array(freq)

    fisher = estimator.compute_fisher_matrix(params=parameters, data=data)

    print(fisher)

    crlb = estimator.compute_crlb(fisher)

    print("CRLB", crlb)

    var_A, var_v0, var_alpha = simulator.get_variances(folder)

    calculated = np.array([var_A, var_v0, var_alpha])

    print("Calculated", calculated)


def ex_5():
    """
    Example 5: Performs Bayesian Analysis to compute
    marginalized probability density functions (PDFs) and generates
    contour plots for the parameter relationships,
    as well as marginalized PDFs for individual parameters.
    """
    likelihood = ModelConstructor().get_not_normalized_log_likelihood()

    A_min, A_max = tuple((4, 5))
    v0_min, v_max = tuple((0.95, 1.05))
    alpha_min, alpha_max = tuple((0.625, 0.740))

    A = np.linspace(A_min, A_max, 100)
    v0 = np.linspace(v0_min, v_max, 100)
    alpha = np.linspace(alpha_min, alpha_max, 100)

    analysist = BayesianAnalysis(nu, s_n)

    (
        marginalized_A_v0,  # P(A,νo)
        marginalized_A_alpha,  # P(A,α)
        marginalized_v0_alpha,  # P(νo,α)
        marginalized_A,  # P(A)
        marginalized_v0,  # P(νo)
        marginalized_alpha,  # P(α)
    ) = analysist.compute_pdfs(likelihood, A, v0, alpha)

    plotter = Plotter()

    plotter.contour_plot(
        A,
        v0,
        marginalized_A_v0,
        "Amplitude (A)",
        "v0",
        "Marginalized (over alpha) Likelihood\n P(A,νo)",
    )

    plotter.contour_plot(
        A,
        alpha,
        marginalized_A_alpha,
        "Amplitude (A)",
        "alpha",
        "Marginalized (over v0) Likelihood\n P(A,α)",
    )

    plotter.contour_plot(
        v0,
        alpha,
        marginalized_v0_alpha,
        "v0",
        "alpha",
        "Marginalized (over A) Likelihood\n P(νo,α)",
    )

    plotter.marginalized_pdf(
        A,
        marginalized_A,
        "Amplitude (A)",
        "Marginalized PDF of the amplitude A\n P(A)",
        "P(A)",
    )

    plotter.marginalized_pdf(
        v0, marginalized_v0, "v0", "Marginalized PDF of v0\n P(νo)", "P(νo)"
    )

    plotter.marginalized_pdf(
        alpha,
        marginalized_alpha,
        "alpha",
        "Marginalized PDF of alpha\n P(α)",
        "P(α)",
    )


def ex_6(folder: str = "results1_last"):
    """
    Example 6: Uses the Metropolis-Hastings algorithm to
    perform Markov Chain Monte Carlo (MCMC) sampling for estimating the
    posterior distribution of parameters. Generates plots for the
    sampled parameters.
    """
    constructor = ModelConstructor()

    likelihood = constructor.get_likelihood()
    multivariate_gaussian = constructor.get_multivariate_gaussian()

    simulator = MCSimulator(freq, signal)

    (var_A, var_v0, var_alpha) = simulator.get_variances(folder)

    mg_kwargs = {"covariance_matrix": 3 * np.diag([var_A, var_v0, var_alpha])}

    initial_sample = np.array(INITIAL_GUESS)
    data = tuple((freq, signal_noise))

    mcmc = MCMC(likelihood, multivariate_gaussian, **mg_kwargs)

    samples = mcmc.metropolis_hastings(initial_sample, data)

    print("Samples len:", len(samples))

    samples_array = np.vstack(samples)

    A_vals = samples_array[:, 0]
    A_mean = A_vals.mean()
    A_std = A_vals.std()

    v_0_vals = samples_array[:, 1]
    v_0_mean = v_0_vals.mean()
    v_0_std = v_0_vals.std()

    alpha_vals = samples_array[:, 2]
    alpha_mean = alpha_vals.mean()
    alpha_std = alpha_vals.std()

    plotter = Plotter()

    plotter.plot_estimated_parameter(
        A_vals,
        "MCMC samples of A \n",
        "Amplitude (A)",
        A_mean,
        A_std,
    )
    plotter.plot_estimated_parameter(
        v_0_vals,
        "MCMC samples of v0 \n",
        "v_0",
        v_0_mean,
        v_0_std,
    )
    plotter.plot_estimated_parameter(
        alpha_vals,
        "MCMC samples of alpha \n",
        "alpha",
        alpha_mean,
        alpha_std,
    )
