from typing import Callable, Any
import numpy as np
from tqdm import tqdm


class MCMC:
    """
    A class to perform Markov Chain Monte Carlo (MCMC) sampling
    using the Metropolis-Hastings algorithm.
    """

    def __init__(
        self,
        likelihood: Callable[..., Any],
        proposed_distribution: Callable[..., Any],
        **kwargs_proposal_dist,
    ):
        """
        Initializes the MCMC class.

        :param likelihood: A function that evaluates the likelihood
        for a given set of parameters and data.
        :type likelihood: Callable[..., Any]
        :param proposed_distribution: A function that generates trial
        samples for the proposal distribution.
        :type proposed_distribution: Callable[..., Any]
        :param kwargs_proposal_dist: Additional arguments to pass to
        the proposal distribution function.
        :type kwargs_proposal_dist: dict
        """
        self._likelihood = likelihood
        self._proposed_distribution = proposed_distribution
        self._proposad_distribution_kwargs = kwargs_proposal_dist

    def metropolis_hastings(
        self,
        initial_sample: np.ndarray,
        data: tuple[np.ndarray, np.ndarray],
        num: int = 100000,
        burn_in: int = 1000,
    ):
        """
        Executes the Metropolis-Hastings algorithm for MCMC sampling.

        :param initial_sample: The initial parameter values for the chain.
        :type initial_sample: np.ndarray
        :param data: The observed data used in the likelihood function.
        :type data: tuple[np.ndarray, np.ndarray]
        :param num: The total number of samples to generate
        (default is 100000).
        :type num: int, optional
        :param burn_in: The number of initial samples to discard as burn-in
        (default is 1000).
        :type burn_in: int, optional
        :return: The retained samples after the burn-in period.
        :rtype: list[np.ndarray]
        """
        samples = [initial_sample]

        num_accepted_samples = 0
        num_total_samples = 0

        for _ in tqdm(range(num), desc="Sampling"):
            trial_sample = self._proposed_distribution(
                samples[-1], **self._proposad_distribution_kwargs
            )
            trial_sample = np.array(trial_sample[0])

            a = self._likelihood(
                parameters=trial_sample, data=data
            ) / self._likelihood(parameters=samples[-1], data=data)

            if a > 1:
                samples.append(trial_sample)
                num_accepted_samples += 1
            else:
                rand_num = np.random.random()
                if rand_num < a:
                    samples.append(trial_sample)
                    num_accepted_samples += 1
                else:
                    samples.append(samples[-1])

            num_total_samples += 1

        self._print_analytics(num_accepted_samples, num_total_samples)

        retained_samples = samples[burn_in:]

        return retained_samples

    def _print_analytics(
        self, num_accepted_samples: int, num_total_samples: int
    ):
        """
        Prints summary statistics of the sampling process.

        :param num_accepted_samples: The number of samples
        accepted during sampling.
        :type num_accepted_samples: int
        :param num_total_samples: The total number of
        samples attempted.
        :type num_total_samples: int
        """
        acceptance_rate = (num_accepted_samples / num_total_samples) * 100
        header = "MCMC - Metropolis Hastings analytics"
        line_length = 49

        def format_line(content: str) -> str:
            """
            Formats a string to be left-justified within a
            fixed-width space and wraps it with vertical bars.

            :param content: The text to format.
            :type content: str
            :return: The formatted string, padded with spaces
            and wrapped in vertical bars.
            :rtype: str
            """
            return f"| {content.ljust(line_length - 4)} |"

        print("-" * line_length)
        print(format_line(header.center(line_length - 4)))
        print("-" * line_length)
        print(
            format_line(f"Number of accepted samples = {num_accepted_samples}")
        )
        print(format_line(f"Number of total samples = {num_total_samples}"))
        print("-" * line_length)
        print(format_line(f"Acceptance rate = {acceptance_rate:.2f} %"))
        print("-" * line_length)
