import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp
from scipy.stats import norm


class Plotter:
    def plot_model_function(
        self, freq: list[float], signal: list[float], signal_noise: list[float]
    ) -> None:
        """
        Plots the model function and the signal with white
        gaussian noise (WGN).

        :param freq: list with frequencies
        :type freq: list[float]
        :param signal: list with the signal
        :type signal: list[float]
        :param signal_noise: list with the signal + noise
        :type signal_noise: list[float]
        """
        _, ax = plt.subplots()

        ax.plot(freq, signal, label="Signal", color="blue")
        ax.scatter(
            freq, signal_noise, label="Signal + WGN", color="black", marker="*"
        )

        ax.set_xlabel("frequency (ν)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Signal vs frequency")
        ax.legend()
        plt.show()

    def plot_mc_realizations(
        self,
        nu: jnp.ndarray,
        signal: jnp.ndarray,
        realizations: np.ndarray,
        num_realizations: int = 5,
    ) -> None:
        """
        Makes a 3D plot of the realizations of the signal
        after the Monty Carlo.

        :param nu: the frequency
        :type nu: jnp.ndarray
        :param signal: the signal
        :type signal:: jnp.ndarray
        :param realizations: the signal+noise realizations
        :type realizations:: np.ndarray
        :param num_realizations: the number of realizations to plot,
        defaults to 5
        :type num_realizations: int, optional
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot each realization with a y-offset
        for i in range(num_realizations):
            y = np.full_like(signal, i)  # y-axis offset for spacing
            z = realizations[i]  # z-axis: signal + noise values

            ax.scatter(nu, y, z, label=f"Realization {i+1}", s=50)

        # Plot the signal as first
        y_signal = -1
        ax.plot(
            nu,
            np.full_like(signal, y_signal),
            signal,
            color="black",
            linewidth=2,
            label="Original Signal",
        )

        ax.set_title(
            f"3D Plot of Signal with Noise (First {num_realizations} Realizations)"
        )
        ax.set_xlabel("Nu")
        ax.set_ylabel("Realization")
        ax.set_zlabel("Value")
        ax.view_init(elev=20, azim=-60)
        ax.legend()

        plt.show()

    def plot_estimated_parameter(
        self, data: np.ndarray, title: str, x_labels: str
    ):
        mean = data.mean()
        std = data.std()

        self._plot_hist(data, title, x_labels, mean, std)

        return mean, std

    def _plot_hist(
        self,
        data: np.ndarray,
        title: str,
        x_labels: str,
        mean: float,
        std: float,
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

    def plot_estimates_with_variance(self, estimates: np.ndarray):
        """
        Plots the mean and variance (as a shaded area) of the estimates.
        """
        # Calculate the mean and standard deviation across runs (axis=0)
        mean_estimates = np.mean(
            estimates, axis=0
        )  # Shape: (steps, num_params)
        std_estimates = np.std(estimates, axis=0)  # Shape: (steps, num_params)

        # Extract mean and std for each parameter
        mean_A, mean_v_0, mean_alpha = mean_estimates.T
        std_A, std_v_0, std_alpha = std_estimates.T

        # Get the number of steps
        steps = np.arange(len(mean_A))

        # Plotting
        plt.figure(figsize=(10, 8))

        # A
        plt.plot(steps, mean_A, label="Mean A", color="blue")
        plt.fill_between(
            steps, mean_A - std_A, mean_A + std_A, color="blue", alpha=0.2
        )

        # v_0
        plt.plot(steps, mean_v_0, label="Mean v_0", color="green")
        plt.fill_between(
            steps,
            mean_v_0 - std_v_0,
            mean_v_0 + std_v_0,
            color="green",
            alpha=0.2,
        )

        # alpha
        plt.plot(steps, mean_alpha, label="Mean alpha", color="red")
        plt.fill_between(
            steps,
            mean_alpha - std_alpha,
            mean_alpha + std_alpha,
            color="red",
            alpha=0.2,
        )

        # Labels and legend
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(
            "Mean and Variance of Parameter Estimates\n duiring the Newton-Raphson optimization"
        )
        plt.grid(True)
        plt.legend()
        plt.show()
