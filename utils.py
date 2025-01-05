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

def save(all_estimates, estimates, gradient_norms, count):
    # Save the dictionary to a file
    with open(f"all_estimates_{count+8000}_last.pkl", "wb") as file:
        pickle.dump(all_estimates, file)

    with open(f"estimates_{count+8000}_last.pkl", "wb") as file:
        pickle.dump(estimates, file)

    with open(f"gradient_norms_{count+8000}_last.pkl", "wb") as file:
        pickle.dump(gradient_norms, file)