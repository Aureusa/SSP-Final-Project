from typing import Callable, Any
import jax.numpy as jnp


class ModelConstructor:
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
        return self._generative_model

    def _generative_model(self, v_i, A, v_0, alpha):
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
