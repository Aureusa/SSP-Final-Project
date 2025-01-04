from utils import get_data
import jax.numpy as jnp
import pickle

from estimator.estimator import MLEGeneralJAX
from model.model_constructor import ModelConstructor
from markov_chain_monte_carlo.mc_generator import MCGenerator
from plotting.plotting_utils import Plotter

freq, signal, signal_noise = get_data()
nu = jnp.array(freq.tolist())
s_n = jnp.array(signal_noise.tolist())
DATA = tuple((nu, s_n))

INITIAL_GUESS = [6.0, 2.0, 1.0]


def mle_general_jax():
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

    # mle_estimator.plot_gradient_norms()
    # mle_estimator.plot_estimates()
    # mle_estimator.plot_residuals(guess)

    print(guess)

def ex_3(plot: bool = False):
    # Get the models used by the MLE
    model_constructor = ModelConstructor()
    log_likelihood = model_constructor.get_log_likelihood()
    generative_model = model_constructor.get_generative_model()

    # Define the nuisance parameters
    nuisance_params = ["v", "s_n", "sigma"]

    # Instantiate the MLE estimator
    mle_estimator = MLEGeneralJAX(
        model=generative_model,
        log_likelihood=log_likelihood,
        nuisance_params=nuisance_params,
        data=None,
        gamma=0.1,
    )

    # Get the frequencies
    nu = jnp.array(freq.tolist())

    # Generate realizations
    signal_generator = MCGenerator(signal)
    realizations = signal_generator.generate_mc_realizations()

    if plot:
        plotter = Plotter()
        plotter.plot_mc_realizations(nu, signal, realizations)

    # Initialise a dict containing all the estimates
    all_estimates = {"A": [], "v_0": [], "alpha": []}
    
    estimates: list[list[float]] = []
    gradient_norms: list[list[float]] = []
    count = 0
    for realiz in realizations:
        count += 1
        print(count)
        if count == 4:
            break
        realiz = jnp.array(realiz.tolist())
        data = tuple((nu, realiz))

        mle_estimator.data = data

        estimate = mle_estimator.evaluate_the_ML_estimate(
            guess=INITIAL_GUESS, data_logging=True, treshhold=1e-6
        )
        print(estimate)
        all_estimates["A"].append(estimate["A"])
        all_estimates["v_0"].append(estimate["v_0"])
        all_estimates["alpha"].append(estimate["alpha"])

        estimates.append(mle_estimator.estimates)
        gradient_norms.append(mle_estimator.gradient_norms)

    # Save the dictionary to a file
    with open("all_estimates.pkl", "wb") as file:
        pickle.dump(all_estimates, file)

    with open("estimates.pkl", "wb") as file:
        pickle.dump(estimates, file)

    with open("gradient_norms.pkl", "wb") as file:
        pickle.dump(gradient_norms, file)
    
    

if __name__ == "__main__":
    ex_3()
