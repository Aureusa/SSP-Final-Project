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


multiprocessing.set_start_method("spawn", force=True)


def process_realization(
    realiz: np.ndarray, nu: jnp.ndarray, mle_estimator_params: dict
) -> dict:
    """
    Processes a single realization in parallel using an MLE estimator.

    :param realiz: A realization of the data to be processed
    :type realiz: np.ndarray
    :param nu: Frequencies associated with the data
    :type nu: jnp.ndarray
    :param mle_estimator_params: Parameters for initializing
    the MLE estimator
    :type mle_estimator_params: dict
    :return: A dictionary containing the estimated parameters
    and diagnostic information
    :rtype: dict
    """
    # Import inside to not get issues with multiprocessing
    from estimator.estimator import MLEGeneralJAX

    # Reinstantiates MLEGeneralJAX for each CPU
    mle_estimator = MLEGeneralJAX(
        model=mle_estimator_params["model"],
        log_likelihood=mle_estimator_params["log_likelihood"],
        nuisance_params=mle_estimator_params["nuisance_params"],
        data=None,
        gamma=mle_estimator_params["gamma"],
    )

    # Preps the data
    realiz = jnp.array(realiz.tolist())
    data = tuple((nu, realiz))

    # Set the data of the estimator using the setter
    mle_estimator.data = data

    # Perform the estimation
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
    def __init__(
        self,
        freq: list[float],
        signal: list[float],
        realizations_filename: str = "realizations.pkl",
    ) -> None:
        """
        Initializes the MCSimulator instance with frequency
        ]and signal data.

        :param freq: A list of frequencies
        :type freq: list[float]
        :param signal: A list representing the signal
        :type signal: list[float]
        :param realizations_filename: Filename for storing or
        loading realizations, defaults to "realizations.pkl"
        :type realizations_filename: str, optional
        """
        self._freq = freq
        self._signal = signal
        self._realization_filepath = os.path.join(
            REALIZATIONS_FOLDER, realizations_filename
        )

    def perform_simulation(self, num_cpus: int = 8) -> None:
        """
        Runs a Monte Carlo simulation in parallel to
        process realizations.

        :param num_cpus: Number of CPU cores to use for
        parallel processing, defaults to 8
        :type num_cpus: int, optional
        """
        print("=============== Simulation started ===============")
        # Get the models used by the MLE
        model_constructor = ModelConstructor()
        log_likelihood = model_constructor.get_log_likelihood()
        generative_model = model_constructor.get_generative_model()

        # Define the nuisance parameters
        nuisance_params = ["v", "s_n", "sigma"]

        # Define the MLEGeneralJAX's parameters for instantiation
        mle_estimator_params = {
            "model": generative_model,
            "log_likelihood": log_likelihood,
            "nuisance_params": nuisance_params,
            "gamma": 0.1,
        }

        # Get the frequencies
        nu = jnp.array(self._freq.tolist())

        # Get MC realizations of the data
        realizations = self._get_realizations()

        # Initialise a dict containing all the estimates to be logged
        all_estimates = {"A": [], "v_0": [], "alpha": []}
        estimates: List[List[float]] = []
        gradient_norms: List[List[float]] = []

        # Allocate tasks to the different CPUs
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_cpus
        ) as executor:
            futures = [
                executor.submit(
                    process_realization, realiz, nu, mle_estimator_params
                )
                for realiz in realizations
            ]

            # Collect results as the come
            for count, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                print(f"{count}/{len(realizations)}")
                result = future.result()

                # Update the dictionaries and lists
                all_estimates["A"].append(result["A"])
                all_estimates["v_0"].append(result["v_0"])
                all_estimates["alpha"].append(result["alpha"])
                estimates.append(result["estimates"])
                gradient_norms.append(result["gradient_norms"])

                # Save results at CHECKPOINTS in case of a crash
                if count in CHECKPOINTS:
                    save(all_estimates, estimates, gradient_norms, count)

        # Save everything at the end
        save(all_estimates, estimates, gradient_norms, "ALL")

    def plot_mc_realizations(self, num_realizations: int = 5) -> None:
        """
        Plots Monte Carlo realizations of the signal.

        :param num_realizations: Number of realizations to plot,
        defaults to 5
        :type num_realizations: int, optional
        """
        # Get the frequencies
        nu = jnp.array(self._freq.tolist())

        # Get realizations
        realizations = self._get_realizations()

        plotter = Plotter()
        plotter.plot_mc_realizations(
            nu, self._signal, realizations, num_realizations
        )

    def _get_realizations(self) -> np.ndarray:
        """
        Loads or generates Monte Carlo realizations.

        :return: An array of realizations
        :rtype: np.ndarray
        """
        if os.path.exists(self._realization_filepath):
            # If the file exists, load the content
            with open(self._realization_filepath, "rb") as file:
                realizations: np.ndarray = pickle.load(file)
            print("Loaded realizations from file.")
            realizations = realizations
        else:
            # Generate realizations if file not found
            print(
                f"No such filepath exists: {self._realization_filepath}\n "
                "Generating realiaztions..."
            )
            signal_generator = MCGenerator(self._signal)
            realizations = signal_generator.generate_mc_realizations()

            new_file = os.path.join(REALIZATIONS_FOLDER, "realizations.pkl")
            with open(new_file, "wb") as file:
                pickle.dump(realizations, file)
            print(
                "Generation complete! The realizations can be found in the following dir:\n",
                new_file,
            )

        return realizations

    def plot_estimates_pdf(self, folder) -> None:
        """
        Plots the probability density function (PDF) of the
        estimated parameters.

        :param filename: The filename containing the estimates
        :type filename: str, optional
        """
        plotter = Plotter()

        data, mean_A, var_A, mean_v0, var_v0, mean_alpha, var_alpha = (
            self._analyse_data(folder)
        )

        plotter.plot_estimated_parameter(
            np.array(data["A"]), "Parameter A", "A vals", mean_A, var_A**0.5
        )
        plotter.plot_estimated_parameter(
            np.array(data["v_0"]),
            "Parameter v_0",
            "v_0 vals",
            mean_v0,
            var_v0**0.5,
        )
        plotter.plot_estimated_parameter(
            np.array(data["alpha"]),
            "Parameter alpha",
            "alpha vals",
            mean_alpha,
            var_alpha**0.5,
        )

    def _analyse_data(
        self, folder
    ) -> tuple[dict, float, float, float, float, float, float]:
        """
        Analyzes data from a specified file,
        calculating mean and variance for
        parameters A, v_0, and alpha.

        :param filename: The name of the file
        containing the data to be analyzed.
        :type filename: str
        :return: A tuple containing:
            - data (dict): The loaded data from the file.
            - mean_A (float): The mean value of parameter A.
            - var_A (float): The variance of parameter A.
            - mean_v0 (float): The mean value of parameter v_0.
            - var_v0 (float): The variance of parameter v_0.
            - mean_alpha (float): The mean value of parameter alpha.
            - var_alpha (float): The variance of parameter alpha.
        :rtype: tuple[dict, float, float, float, float, float, float]
        """
        results_folder = os.path.join(REALIZATIONS_FOLDER, folder)
        filepath = os.path.join(results_folder, "all_estimates.pkl")
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        mean_A = np.array(data["A"]).mean()
        var_A = np.array(data["A"]).std() ** 2

        mean_v0 = np.array(data["v_0"]).mean()
        var_v0 = np.array(data["v_0"]).std() ** 2

        mean_alpha = np.array(data["alpha"]).mean()
        var_alpha = np.array(data["alpha"]).std() ** 2

        return data, mean_A, var_A, mean_v0, var_v0, mean_alpha, var_alpha

    def get_means(self, folder) -> tuple[float, float, float]:
        """
        Computes the means of the parameters (A, v_0, alpha)
        from the analysis of the data.

        :param filename: The folder of the results with the
        data to analyze.
        :type filename: str, optional
        :return: The mean values of A, v_0, and alpha.
        :rtype: tuple[float,float,float]
        """
        (_, mean_A, _, mean_v0, _, mean_alpha, _) = self._analyse_data(folder)
        return mean_A, mean_v0, mean_alpha

    def get_variances(self, folder) -> tuple[float, float, float]:
        """
        Computes the variances of the parameters (A, v_0, alpha)
        from the analysis of the data.

        :param filename: The name of the folder containing the
        data to analyze.
        :type filename: str, optional
        :return: The variances of A, v_0, and alpha.
        :rtype: tuple[float,float,float]
        """
        (_, _, var_A, _, var_v0, _, var_alpha) = self._analyse_data(folder)
        return var_A, var_v0, var_alpha

    def plot_estimates_duiring_nr_optimization(self, folder):
        """
        Plots parameter estimates during Newton-Raphson
        optimization.

        :param filename: The filename containing the optimization
        estimates
        :type filename: str, optional
        """
        # Retrieve the estimates duiring each optimization run
        results_folder = os.path.join(REALIZATIONS_FOLDER, folder)
        filepath = os.path.join(results_folder, "estimates_nr.pkl")
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        padded_data = self._add_padding(data)

        plotter = Plotter()

        plotter.plot_estimates_with_variance(padded_data)

    def _add_padding(self, estimates: list[list[list[float]]]) -> np.ndarray:
        """
        Adds padding to the 3D list of estimates by extending each
        inner list to the maximum length with repetitions of its last value.
        This is done in order to be able to convert this list
        into a numpy object and avoid inhomoguneity error.

        :param data: A 3D list of estimates, where each innermost list represents a run of values
        :type data: list[list[list[float]]]
        :return: A 3D NumPy array with all runs padded to the maximum length
        :rtype: np.ndarray
        """
        # Calculates the length of the longest run
        max_len = max(len(run) for run in estimates)

        # Pad each shorter run by repeating the last value
        padded_data = []
        for run in estimates:
            # Calculate the padding needed
            padding_needed = max_len - len(run)

            if padding_needed > 0:
                run += [run[-1]] * padding_needed

            padded_data.append(run)

        padded_data = np.array(padded_data)

        return padded_data
