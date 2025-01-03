import inspect
from typing import Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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
            model: Callable[...,Any],
            nuisance_params: list[str],
            data: tuple[list],
            gamma: float
        ) -> None:
        """
        Used to instantiate the MLE class.

        :param model: _description_
        :type model: Callable[...,Any]
        :param nu: _description_
        :type nu: list[float]
        :param s_n: _description_
        :type s_n: list[float]
        :param gamma: _description_
        :type gamma: float
        """
        self._log_model = model
        self._nuisance_params = nuisance_params
        self._data = data
        self._gamma = gamma
        self._gradient_norms: list[float] = []
        self._estimates: list[float] = []
        self._params_to_estimate: list[str]|None = None
    
    def evaluate_the_ML_estimate(
            self,
            guess: list[float],
            data_logging: bool,
            treshhold: float
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
                -
                self._gamma * self._compute_gradient_vector_and_jacobian(initial_guess_arr)
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
    
    def _compute_gradient_vector_and_jacobian(
            self,
            vals: jnp.ndarray
        ) -> jnp.ndarray:
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
        model_signature = inspect.signature(self._log_model)
        parameters = list(model_signature.parameters.keys())
        num_parameters = len(parameters)
        argnums = [num for num in range(num_parameters)]
        for _ in range(len(self._nuisance_params)):
            argnums.pop()
        if self._params_to_estimate is None:
            self._params_to_estimate = parameters[:argnums[-1]+1]
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
        hessian_model = jax.hessian(self._log_model, argnums=argnums)
        hessian_model_eval = hessian_model(*values)
        hessian = jnp.array(hessian_model_eval)
        return hessian

    def _compute_gradient(self) -> Callable[...,Any]:
        """
        Computes the gradient.

        :return: the gradient as a function.
        :rtype: Callable[...,Any]
        """
        # Compute the gradient of f with respect to x
        argnums = self._get_argnums()
        grad_model = jax.grad(self._log_model, argnums=argnums)
        return grad_model
    