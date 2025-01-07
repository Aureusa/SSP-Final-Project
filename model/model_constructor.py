from typing import Callable, Any
import jax.numpy as jnp
import numpy as np


class ModelConstructor:
    """
    A class for constructing models, including likelihood functions,
    generative models, and multivariate Gaussian sampling.
    """

    def get_multivariate_gaussian(self) -> Callable[..., Any]:
        """
        Gets the multivariate Gaussian sampling function.

        :return: The multivariate Gaussian function.
        :rtype: Callable[...,Any]
        """
        return self._multivariate_gaussian

    def get_likelihood(self) -> Callable[..., Any]:
        """
        Gets the likelihood function.

        :return: The likelihood function.
        :rtype: Callable[...,Any]
        """

        return self._likelihood

    def _likelihood(
        self, parameters: np.ndarray, data: tuple[np.ndarray, np.ndarray]
    ) -> float:
        """
        Computes the likelihood for the given parameters and data.

        :param parameters: Model parameters (A, v_0, alpha).
        :type parameters: np.ndarray
        :param data: Tuple containing `v_i` (velocity data) and `s_n` (signal data).
        :type data: tuple[np.ndarray, np.ndarray]
        :return: The likelihood value.
        :rtype: float
        """
        v_i, s_n = data

        A = parameters[0]
        v_0 = parameters[1]
        alpha = parameters[2]

        model = A * (v_i / v_0) ** alpha * (1 + v_i / v_0) ** (-4 * alpha)
        return np.exp(np.sum(-((s_n - model) ** 2)))

    def _multivariate_gaussian(
        self, mean: np.ndarray, covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Generates a sample from a multivariate Gaussian distribution.

        :param mean: Mean of the Gaussian distribution.
        :type mean: np.ndarray
        :param covariance_matrix: Covariance matrix of the
        Gaussian distribution.
        :type covariance_matrix: np.ndarray
        :return: A sample from the Gaussian distribution.
        :rtype: np.ndarray
        """
        return np.random.multivariate_normal(mean, covariance_matrix, 1)

    def get_log_likelihood(self) -> Callable[..., Any]:
        """
        Gets the functon for the model.

        :return: the log likelihood model.
        :rtype: Callable[...,Any]
        """
        return self._log_likelihood

    def get_not_normalized_log_likelihood(self) -> Callable[..., Any]:
        """
        Gets the functon for the model.

        :return: the log likelihood model.
        :rtype: Callable[...,Any]
        """
        return self._not_normalized_log_likelihood

    def get_generative_model(self) -> Callable[..., Any]:
        """
        Gets the generative model function.

        :return: The generative model function.
        :rtype: Callable[..., Any]
        """
        return self._generative_model

    def _generative_model(
        self, v_i: np.ndarray, A: float, v_0: float, alpha: float
    ) -> np.ndarray:
        """
        Computes the generative model output for given inputs.

        :param v_i: Input velocity values.
        :type v_i: np.ndarray
        :param A: Amplitude parameter.
        :type A: float
        :param v_0: Scale parameter.
        :type v_0: float
        :param alpha: Shape parameter.
        :type alpha: float
        :return: The generative model output.
        :rtype: np.ndarray
        """
        frac = v_i / v_0
        y_val = A * frac**alpha * (1 + frac) ** (-4 * alpha)
        return y_val

    def _log_likelihood(
        self,
        A: float,
        v_0: float,
        alpha: float,
        v_i: float,
        s_n: list[float],
        sigma: float = jnp.sqrt(0.0025),
    ) -> float:
        """
        Defines the model used to generate data and calculates the negative log-likelihood
        based on observed data and the model.

        :param A: Amplitude of the model.
        :type A: float
        :param v_0: Reference frequency or baseline frequency.
        :type v_0: float
        :param alpha: Power-law index or scaling exponent.
        :type alpha: float
        :param v_i: Frequency at which the data point is generated.
        :type v_i: float
        :param s_n: List of observed data points (signal+noise).
        :type s_n: list[float]
        :param sigma: Standard deviation of the noise, default is sqrt(0.0025).
        :type sigma: float
        :return: Negative log-likelihood of the model given the observed data.
        :rtype: float
        """
        model = A * (v_i / v_0) ** alpha * (1 + v_i / v_0) ** (-4 * alpha)
        return jnp.sum(-0.5 * (s_n - model) ** 2 / sigma**2)

    def _not_normalized_log_likelihood(
        self,
        A: float,
        v_0: float,
        alpha: float,
        v_i: float,
        s_n: list[float],
        sigma: float = jnp.sqrt(0.0025),
    ) -> float:
        """
        Defines the model used to generate data and calculates the negative log-likelihood
        based on observed data and the model.

        :param A: Amplitude of the model.
        :type A: float
        :param v_0: Reference frequency or baseline frequency.
        :type v_0: float
        :param alpha: Power-law index or scaling exponent.
        :type alpha: float
        :param v_i: Frequency at which the data point is generated.
        :type v_i: float
        :param s_n: List of observed data points (signal+noise).
        :type s_n: list[float]
        :param sigma: Standard deviation of the noise, default is sqrt(0.0025).
        :type sigma: float
        :return: Negative log-likelihood of the model given the observed data.
        :rtype: float
        """
        model = A * (v_i / v_0) ** alpha * (1 + v_i / v_0) ** (-4 * alpha)
        return jnp.sum(-((s_n - model) ** 2))
