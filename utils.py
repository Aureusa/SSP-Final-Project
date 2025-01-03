import os
import pandas as pd
import numpy as np
from typing import Any, Callable

def get_data() -> tuple[list[float],list[float],list[float]]:
    """
    Gets the data from the csv file.

    :return: a tuple containing the data in
    (frequency, signal, signal+noise)
    :rtype: tuple[list[float],list[float],list[float]]
    """
    data_dir = os.path.join(os.getcwd(), "data")
    filename = os.path.join(data_dir, "project_data.csv")
    data = pd.read_csv(filename)
    freq = data["freq"]
    signal = data["Signal"]
    signal_noise = data["Signal+Noise"]
    return freq, signal, signal_noise

def model(
        amp: float,
        v_0: float,
        alpha: float,
        v_i: float,
        wgn: float
        ) -> float:
    """
    Defines the model that is used to generate the data.

    :param amp: the amplitude
    :type amp: float
    :param v_0: the estimated frequency
    :type v_0: float
    :param alpha: the estimated alpha value
    :type alpha: float
    :param v_i: the frequency nu at which
    the datapoint is generated
    :type v_i: float
    :param wgn: the white gaussian noise
    :type wgn: float
    :return: the datapoint at nu
    :rtype: float
    """
    frac = v_i/v_0
    x_val = amp * frac ** alpha * (1 + frac) ** (-4 * alpha) + wgn
    return x_val

def run_monte_carlo(
        model: Callable[...,Any],
        num: int = 1e5,
        nu_range: list = (0, 1),
        **kwargs
        ) -> np.ndarray:
    """
    Generates Monte Carlo realizations of x by adding
    different realizations of noise comming from
    N(σ = 0.05, µ = 0) and using those to produce datapoints
    using a model.

    :param model: the model to use, must be a callable function!
    :type model: Callable[...,Any]
    :param num: the number of realizations to be generated,
    defaults to 1e5
    :type num: int, optional
    :param nu_range: the range of nu, defaults to (0, 1)
    :type nu_range: list, optional
    :param **kwargs: any keyword arguments to be passed to the model
    function. Check the documentation of the model function and pass like:
        Example:
        run_monte_carlo(...,model_key_word=value,...)
    :return: an array of the generated realizations of x at each nu
    :rtype: list[float]
    """
    nu_arr = np.linspace(start=nu_range[0], stop=nu_range[1], num=num)
    x_values = []
    for nu in nu_arr:
        x_values.append(model(**kwargs))
    x_arr = np.array(x_values)
    return x_arr
