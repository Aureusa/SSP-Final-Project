import matplotlib.pyplot as plt


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
