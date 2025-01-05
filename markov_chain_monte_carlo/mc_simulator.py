import jax.numpy as jnp
import numpy as np
import pickle
from typing import List
import os
import concurrent.futures
import multiprocessing

from model.model_constructor import ModelConstructor
from markov_chain_monte_carlo.mc_generator import MCGenerator
from plotting.plotting_utils import Plotter
from utils import save


REALIZATIONS_FOLDER = os.path.join(os.getcwd(), "results")
INITIAL_GUESS = [6.0, 2.0, 1.0]
CHECKPOINTS = list(range(1000, 10001, 1000))


multiprocessing.set_start_method('spawn', force=True)


def process_realization(realiz, nu, mle_estimator_params):
    """
    Function to process a single realization in parallel.
    """
    # Import inside to avoid issues with multiprocessing
    from estimator.estimator import MLEGeneralJAX
    
    # Recreate the MLE estimator in the new process
    mle_estimator = MLEGeneralJAX(
        model=mle_estimator_params["model"],
        log_likelihood=mle_estimator_params["log_likelihood"],
        nuisance_params=mle_estimator_params["nuisance_params"],
        data=None,
        gamma=mle_estimator_params["gamma"],
    )
    
    realiz = jnp.array(realiz.tolist())
    data = tuple((nu, realiz))

    mle_estimator.data = data

    estimate = mle_estimator.evaluate_the_ML_estimate(
        guess=INITIAL_GUESS, data_logging=True, treshhold=1e-6
    )

    return {
        "A": estimate["A"],
        "v_0": estimate["v_0"],
        "alpha": estimate["alpha"],
        "estimates": mle_estimator.estimates,
        "gradient_norms": mle_estimator.gradient_norms,
    }

class MCSimulator:
    def __init__(self, freq, signal, realizations_filename: str = "realizations.pkl"):
        self._freq = freq
        self._signal = signal
        self._realization_filepath = os.path.join(REALIZATIONS_FOLDER, realizations_filename)

    def perform_simulation(self, num_cpus: int = 8):
        print("=============== Simulation started ===============")
        # Get the models used by the MLE
        model_constructor = ModelConstructor()
        log_likelihood = model_constructor.get_log_likelihood()
        generative_model = model_constructor.get_generative_model()

        # Define the nuisance parameters
        nuisance_params = ["v", "s_n", "sigma"]

        mle_estimator_params = {
            "model": generative_model,
            "log_likelihood": log_likelihood,
            "nuisance_params": nuisance_params,
            "gamma": 0.1,
        }

        # Get the frequencies
        nu = jnp.array(self._freq.tolist())

        # Get realizations of the data
        realizations = self._get_realizations()

        # Initialise a dict containing all the estimates
        all_estimates = {"A": [], "v_0": [], "alpha": []}
        estimates: List[List[float]] = []
        gradient_norms: List[List[float]] = []

        # Use ProcessPoolExecutor with the spawn method
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            # Submit tasks to the executor
            futures = [
                executor.submit(process_realization, realiz, nu, mle_estimator_params)
                for realiz in realizations
            ]

            # Collect results as they complete
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                print(f"{count}/{len(realizations)}")
                result = future.result()

                # Update the dictionaries and lists
                all_estimates["A"].append(result["A"])
                all_estimates["v_0"].append(result["v_0"])
                all_estimates["alpha"].append(result["alpha"])
                estimates.append(result["estimates"])
                gradient_norms.append(result["gradient_norms"])

                if count in CHECKPOINTS:
                    save(all_estimates, estimates, gradient_norms, count)

        save(all_estimates, estimates, gradient_norms, "ALL")

    def plot_mc_realizations(self):
        # Get the frequencies
        nu = jnp.array(self._freq.tolist())

        # Get realizations
        realizations = self._get_realizations()

        plotter = Plotter()
        plotter.plot_mc_realizations(nu, self._signal, realizations)

    def _get_realizations(self):
        if os.path.exists(self._realization_filepath):
            # If the file exists, load the content
            with open(self._realization_filepath, "rb") as file:
                realizations: np.ndarray = pickle.load(file)
            print("Loaded realizations from file.")
            realizations = realizations
            print(realizations.shape)
        else:
            # Generate realizations
            signal_generator = MCGenerator(self._signal)
            realizations = signal_generator.generate_mc_realizations()

            new_file = os.path.join(REALIZATIONS_FOLDER, "realizations.pkl")
            with open(new_file, "wb") as file:
                pickle.dump(realizations, file)
        return realizations
    
    def plot_estimates_pdf(self, filename: str = "all_estimates_final.pkl"):
        filepath = os.path.join(REALIZATIONS_FOLDER, filename)
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        plotter = Plotter()

        plotter.plot_estimated_parameter(np.array(data["A"]), "Parameter A", "A vals")
        plotter.plot_estimated_parameter(np.array(data["v_0"]), "Parameter v_0", "v_0 vals")
        plotter.plot_estimated_parameter(np.array(data["alpha"]), "Parameter alpha", "alpha vals")

    def plot_estimates_duiring_nr_optimization(self, filename: str = "estimates_nr_final.pkl"):
        filepath = os.path.join(REALIZATIONS_FOLDER, filename)
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        max_len = max(len(run) for run in data)

        # Pad each run by repeating the last value
        padded_data = []
        for run in data:
            # Calculate how many iterations need to be added
            padding_needed = max_len - len(run)
            
            # If padding is needed, repeat the last value
            if padding_needed > 0:
                run += [run[-1]] * padding_needed
            
            # Add the padded run to the new list
            padded_data.append(run)

        padded_data = np.array(padded_data)

        plotter = Plotter()

        plotter.plot_estimates_with_variance(padded_data)