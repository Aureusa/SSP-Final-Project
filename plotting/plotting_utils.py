import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp


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
        ax.scatter(freq, signal_noise, label="Signal + WGN", color="black", marker="*")

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
            num_realizations: int = 5
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
        ax = fig.add_subplot(111, projection='3d')
        print(num_realizations)
        # Plot each realization with a y-offset
        for i in range(num_realizations):
            y = np.full_like(signal, i)  # y-axis offset for spacing
            z = realizations[i]          # z-axis: signal + noise values
            
            ax.scatter(nu, y, z, label=f"Realization {i+1}", s=50)
        
        # Plot the signal as first
        y_signal = -1
        ax.plot(nu, np.full_like(signal, y_signal), signal, 
                color="black", linewidth=2, label="Original Signal")

        ax.set_title(
            f"3D Plot of Signal with Noise (First {num_realizations} Realizations)"
        )
        ax.set_xlabel("Nu")
        ax.set_ylabel("Realization")
        ax.set_zlabel("Value")
        ax.view_init(elev=20, azim=-60)
        ax.legend()

        plt.show()
