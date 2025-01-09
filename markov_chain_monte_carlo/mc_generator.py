import numpy as np


class MCGenerator:
    def __init__(self, signal: list[float]) -> None:
        """
        Instantiates an Monty Carlo Generator.

        :param signal: the signal
        :type signal: list[float]
        """
        self._signal = np.array(signal)

    def generate_mc_realizations(self, num: int = 1e4) -> np.ndarray:
        """
        Generates Monte Carlo realizations of x by adding
        different realizations of noise comming from
        N(σ = 0.05, µ = 0) and using those to produce datapoints
        using a model.

        :param num: the number of realizations to be generated,
        defaults to 1e5
        :type num: int, optional
        :return: an array of the generated realizations of x
        :rtype: list[float]
        """
        noise = np.random.normal(
            loc=0, scale=0.5, size=(int(num), len(self._signal))
        )
        return self._signal + noise
