import inspect
from typing import Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import norm
from copy import deepcopy


class MLEGeneralJAX:
    """
    Maximum Likelihood Estimator for the three parameters.
    Numerically evaluate the ML estimate of the parameters,
    using the Newton Raphson’s optimization method. The initial
    guess for the parameters (A, v_-, alpha) = (6.0, 2.0, 1.0).
    The class also provides a way to plot the residuals of the
    fit and their PDF.
    """

    def __init__(
        self,
        model: Callable[..., Any],
        log_likelihood: Callable[..., Any],
        nuisance_params: list[str],
        data: tuple[list]|None,
        gamma: float,
    ) -> None:
        """
        Used to instantiate an MLE.

        :param model: the generative model
        :type model: Callable[..., Any]
        :param log_likelihood: the log likelihood
        :type log_likelihood: Callable[..., Any]
        :param nuisance_params: the nuisance parameters
        :type nuisance_params: list[str]
        :param data: the data
        :type data: tuple[list]
        :param gamma: the learning rate of the gradient
        descent
        :type gamma: float
        """
        self._model = model
        self._log_likelihood = log_likelihood
        self._nuisance_params = nuisance_params
        self._data = data
        self._gamma = gamma
        self._gradient_norms: list[float] = []
        self._estimates: list[float] = []
        self._params_to_estimate: list[str] | None = None

    @property
    def estimates(self):
        return deepcopy(self._estimates)
    
    @property
    def gradient_norms(self):
        return deepcopy(self._gradient_norms)
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._gradient_norms = []
        self._estimates = []
        self._data = data

    def evaluate_the_ML_estimate(
        self, guess: list[float], data_logging: bool, treshhold: float
    ) -> dict:
        """
        Evaluates the maximum likelihood estimate using an initial guess.
        If data_logging is set to True then the data duiring optimization is
        logged and can be used to plot the progress of the optimization.

        :param guess: a list of the initial guess.
        IMPORTANT:
        Values must be floats!!!
        Example:
        WRONG (X) = [1,2,3]
        RIGHT(+) = [1.0,2.0,3.0]
        :type guess: list[float]
        :param data_logging: whethe or not to log the data. T/F.
        :type data_logging: bool
        :param treshhold: the treshhold value that checks wether or not
        there is significant difference between the new and old estiamate.
        Controls the convergence rate.
        :type treshhold: float
        :return: a dictionary containing the estimates.
        :rtype: dict
        """
        initial_guess_arr = jnp.array(guess)

        while True:
            new_guess_arr: jnp.ndarray = (
                initial_guess_arr
                - self._gamma
                * self._compute_gradient_vector_and_jacobian(initial_guess_arr)
            )

            # Calculating the difference between each guess
            diff = (jnp.abs(new_guess_arr - initial_guess_arr)).max()

            # Change the initial guess to the current guess
            initial_guess_arr = new_guess_arr

            if data_logging:
                self._get_logging_data(initial_guess_arr)

            if diff < treshhold:
                break

        # Converting the results to a dictionary
        results = new_guess_arr.tolist()
        result_dict = dict(zip(self._params_to_estimate, results))

        return result_dict

    def plot_estimates(self):
        """
        Plots the estimates as a function of the iterations
        from the data collected during optimization.
        """
        self._plot_validator()

        A, v_0, alpha = np.array(self._estimates).T
        plt.figure(figsize=(8, 6))
        plt.plot(A, label="A")
        plt.plot(v_0, label="v_0")
        plt.plot(alpha, label="alpha")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("Change in parameter estimates with each step.")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_gradient_norms(self) -> None:
        """
        Used to plot the norms of the gradient as a function of iterations
        from the data collected during optimization.
        """
        self._plot_validator()

        plt.figure(figsize=(8, 6))
        plt.plot(self._gradient_norms, label="Gradient Norm")
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm vs Iteration")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_residuals(self, estimates: dict) -> tuple[float, float]:
        """
        Plots the residuals of the modeled and
        measured values, and their PDF.

        :param estimates: A dictionary of estimated
        parameters used for modeling
        :type estimates: dict
        :return: The mean and standard deviation
        of the residuals
        :rtype: tuple[float,float]
        """
        modeled_values = self._model(self._data[0], **estimates)
        measured_values = self._data[1]

        residuals = measured_values - modeled_values

        mean, std = self._analyse_residuals(residuals)

        self._plot_hist(
            data=residuals,
            title="Plot of the residuals and their PDF",
            x_labels="Residual values",
            mean=mean,
            std=std,
        )

        return mean, std

    def _plot_hist(
        self, data: np.ndarray, title: str, x_labels: str, mean: float, std: float
    ) -> None:
        """
        Plots a histogram of the data.

        :param data: the data
        :type data: np.ndarray
        :param title: the title
        :type title: str
        :param x_labels: the x labels
        :type x_labels: str
        """
        plt.figure()

        # Calculates the bin count using Scott's rule
        bin_count = self._get_bin_count(data)

        # Plots the hist and gets the bins
        _, bins, _ = plt.hist(
            data, bins=bin_count, color="blue", edgecolor="black", label="Bins"
        )

        # Generate x values across the histogram range
        x_min, x_max = bins[0], bins[-1]
        x_vals_gauss = np.linspace(x_min, x_max, 1000)

        # Compute Gaussian fit
        gauss = (
            norm.pdf(x_vals_gauss, loc=mean, scale=std)
            * len(data)
            * (bins[1] - bins[0])
        )

        # Plot Gaussian
        plt.plot(
            x_vals_gauss,
            gauss,
            color="red",
            linewidth=2,
            label=f"Gaussian Fit\nmean = {mean:.2f}\nstd = {std:.2f}",
        )

        plt.title(title)
        plt.xlabel(x_labels)
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend()
        plt.show()

    def _get_bin_count(self, data: np.ndarray) -> int:
        """
        Uses the Scott's rule to calculate the
        number of bins. The Scott's rule is defined as:
        k = n ^ (1/3)

        :param data: the data
        :type data: np.ndarray
        :return: the number of bins required as given by Scott's rule.
        :rtype: int
        """
        n = len(data)
        k = round(n ** (1 / 3))
        return int(k)

    def _analyse_residuals(self, residuals: jnp.ndarray) -> tuple[float, float]:
        """
        Analyses the distribution of the residuals. Assumes it is gaussian
        and prints the mean and the standart devation, after which it
        returns them.

        :param residuals: the resisudals
        :type residuals: np.ndarray
        :return: a tuple containing the mean and the standard devaition in this order
        :rtype: tuple[float,float]
        """
        mean = float(residuals.mean())
        std = float(residuals.std())

        # Round the results
        mean = round(mean, 2)
        std = round(std, 2)

        print(f"The mean of the residuals is (rounded to two decimals): {mean:.2f}")
        print(
            f"The standard deviation of the residuals is (rounded to two decimals): {std:.2f}"
        )
        return mean, std

    def _plot_validator(self):
        """
        Validates whether or not plotting is possible by checking whether or not
        the object has used the `evaluate_the_ML_estimate` with `data_logging=True`
        first.
        """
        if len(self._gradient_norms) < 1 or len(self._estimates) < 1:
            raise ValueError(
                "There is no data collected for the gradient norms. You can not plot! "
                "Run the method `evaluate_the_ML_estimate` with `data_logging=True` first "
                "to collect the neccessary data!"
            )

    def _get_logging_data(self, guess_arr: jnp.ndarray) -> None:
        """
        Gets the logging of the data duiting fitting. Collects information about
        the norm of the gradient vector and the estimates. This data is used to
        how the optimization process progesses through each itteration.

        :param guess_arr: the array of the current guess from which to extract
        the relevant information.
        :type guess_arr: jnp.ndarray
        """
        # Compute the norm of the gradient vector
        values = tuple(guess_arr.tolist())
        grad_vec = self._compute_gradient_vector(values)
        grad_norm = jnp.linalg.norm(grad_vec)

        # Append the norm of the gradient vecotor and the estimates
        self._estimates.append(guess_arr.tolist())
        self._gradient_norms.append(grad_norm)

    def _compute_gradient_vector_and_jacobian(self, vals: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient vector and multiplies it with the jacobian matrix
        (the Hessian).

        :param vals: the values used to compute the gradient and the hessian.
        :type vals: jnp.ndarray
        :return: the result from the matrix operation.
        :rtype: jnp.ndarray
        """
        values = tuple(vals.tolist())
        grad_vec = self._compute_gradient_vector(values)
        jacobian_matrix = self._compute_hessian(values)
        result = jnp.linalg.solve(jacobian_matrix, grad_vec)
        return result

    def _get_argnums(self) -> tuple[int]:
        """
        Gets the argnums to be used to evaluate the derivatives.
        First it inspects the signature of the model used to instantiate the
        object. After extracting the signature it pops the last N elements,
        where N is the number of nuisance parameters.
        IMPORTANT:
        The model function needs to be defined in such a way as to have
        the nuisance parameters as the last args.

        :return: a tuple containing the argnums, to be used in differentiation.
        :rtype: tuple[int]
        """
        model_signature = inspect.signature(self._log_likelihood)
        parameters = list(model_signature.parameters.keys())
        num_parameters = len(parameters)
        argnums = [num for num in range(num_parameters)]
        for _ in range(len(self._nuisance_params)):
            argnums.pop()
        if self._params_to_estimate is None:
            self._params_to_estimate = parameters[: argnums[-1] + 1]
        return tuple(argnums)

    def _compute_gradient_vector(self, vals: tuple) -> jnp.ndarray:
        """
        Evaluates the gradient vector based on a tuple of values.

        :param vals: the values to compute the gradient with
        :type vals: tuple
        :return: the row vector representation of the
        gradient vector
        :rtype: jnp.ndarray
        """
        grad = self._compute_gradient()

        # Prepare the values to be used in evaluation
        values = vals + self._data

        gradient_vec = grad(*values)
        vector = jnp.array(gradient_vec).T
        return vector

    def _compute_hessian(self, vals: tuple) -> jnp.ndarray:
        """
        Computes the hessian matrix.

        :param vals: the values to be used in the computation
        :type vals: tuple
        :return: the hessian matrix
        :rtype: jnp.ndarray
        """
        # Compute the Hessian of _log_model with respect to the parameters
        argnums = self._get_argnums()
        values = vals + self._data
        hessian_model = jax.hessian(self._log_likelihood, argnums=argnums)
        hessian_model_eval = hessian_model(*values)
        hessian = jnp.array(hessian_model_eval)
        return hessian

    def _compute_gradient(self) -> Callable[..., Any]:
        """
        Computes the gradient.

        :return: the gradient as a function.
        :rtype: Callable[...,Any]
        """
        # Compute the gradient of f with respect to x
        argnums = self._get_argnums()
        grad_model = jax.grad(self._log_likelihood, argnums=argnums)
        return grad_model
