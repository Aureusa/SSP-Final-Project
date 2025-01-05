import os
import pandas as pd
import pickle


def get_data() -> tuple[list[float], list[float], list[float]]:
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


def save(
    all_estimates: dict,
    estimates: list[list[list[float]]],
    gradient_norms: list[list[float]],
    count: int,
):
    """
    Saves multiple dictionaries to files using pickle, appending the count to the filenames.

    :param all_estimates: A dictionary containing all estimates
    :type all_estimates: dict
    :param estimates: A list containing estimates aquired during each optimization run
    :type estimates: list[list[list[float]]]
    :param gradient_norms: A list containing gradient norms of each optimization run
    :type gradient_norms: list[list[float]]
    :param count: An integer used to append to filenames for uniqueness
    :type count: int
    """
    # Save the dictionary to a file
    with open(f"all_estimates_{count+8000}_last.pkl", "wb") as file:
        pickle.dump(all_estimates, file)

    with open(f"estimates_{count+8000}_last.pkl", "wb") as file:
        pickle.dump(estimates, file)

    with open(f"gradient_norms_{count+8000}_last.pkl", "wb") as file:
        pickle.dump(gradient_norms, file)


def print_mc_results(
    mean_A, var_A, mean_v0, var_v0, mean_alpha, var_alpha
) -> None:
    """
    Prints the results of Monte Carlo estimations for three estimators (A, v0, and alpha),
    including their means and variances.

    :param mean_A: The mean value of estimator A
    :type mean_A: float
    :param var_A: The variance of estimator A
    :type var_A: float
    :param mean_v0: The mean value of estimator v0
    :type mean_v0: float
    :param var_v0: The variance of estimator v0
    :type var_v0: float
    :param mean_alpha: The mean value of estimator alpha
    :type mean_alpha: float
    :param var_alpha: The variance of estimator alpha
    :type var_alpha: float
    """
    print("-------------------")
    print(
        f"Estimator A:\nmean = {round(mean_A,3):.3f}\nvariance = {round(var_A,3):.3f}"
    )
    print("-------------------")
    print(
        f"Estimator v0:\nmean = {round(mean_v0,3):.3f}\nvariance = {round(var_v0,3):.3f}"
    )
    print("-------------------")
    print(
        f"Estimator alpha:\nmean = {round(mean_alpha,3):.3f}\nvariance = {round(var_alpha,3):.3f}"
    )
