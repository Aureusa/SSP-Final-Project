# SSP Final Project

This repository contains the code and resources for the SSP Final Project. The project involves various components such as Maximum Likelihood Estimator (MLE), Monty Carlo (MC) simulations,
the Fisher Information Matrix and its releation to the Cramér–Rao lower bound, Bayesian Analysis, Markov Chain Monte Carlo (MCMC), and data visualization.

## Project Structure

- `bayesian_persuasion/`: Code related to Bayesian analysis.
- `data/`: Dataset used in the project.
- `estimator/`: Contains the MLE implimentation with Jax, the Newton-Raphson optimizer, and the numerical evaluation
  of the Fisher Information Matrix as well as the CRLB.
- `graphs_for_report/`: Contains graph for the graded report.
- `markov_chain_monte_carlo/`: Code for generating different realizations of the data, the Monty Carlo simulator, and the
  implimntation of the Metropolis Hastings (MCMC) sampler.
- `model/`: The different model implementations used throughout the analyis.
- `plotting/`: Utility functions for plotting.
- `results/`: Final results, analysis, and the generated realizations of the data used for the Monty Carlo simulations.
- `Final_Project_SSP_report.pdf`: The final project report.
- `Final_Project_SSP_requirements.pdf`: Project requirements.
- `exercises.py`: Helper functions to solve the different requirments of the project.
- `main.py`: Main script to execute the project.
- `requirements.txt`: Required dependencies.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Aureusa/SSP-Final-Project.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the project, execute the `main.py` file:

```bash
python main.py
```

## License

This project is licensed under the "All Rights Reserved" license. You may not copy, modify, distribute, or use any part of this project for any purposes without explicit permission from the author.

All rights to the project code, documentation, and related materials are reserved by the author. This repository has been created for grading purposes only.

