import numpy as np
import jax.numpy as jnp
from typing import Callable, Any
from multiprocessing import Pool


class BayesianAnalysis:
    def __init__(self, nu: jnp.ndarray, s_n: jnp.ndarray) -> None:
        """
        Initializes the BayesianAnalysis class with provided data arrays.

        :param nu: The frequency data for the model.
        :type nu: jnp.ndarray
        :param s_n: The signal + noise data for the model.
        :type s_n: jnp.ndarray
        """
        self._nu = nu
        self._s_n = s_n

    def _compute_log_likelihood_chunk(self, chunk_info: tuple) -> np.ndarray:
        """
        Helper function to compute the log-likelihood for a chunk of the grid.

        This function computes log-likelihood values for each point in a given chunk
        of the model parameter grid (A, v0, alpha).

        :param chunk_info: A tuple containing the chunks of A, v0, alpha grids,
        likelihood function, and the data (nu, s_n).
        :type chunk_info: tuple
        :return: A chunk of log-likelihood values for the grid chunk.
        :rtype: np.ndarray
        """
        A_grid_chunk, v_0_grid_chunk, alpha_grid_chunk, likelihood, nu, s_n = (
            chunk_info
        )
        log_likelihood_values_chunk = np.zeros(A_grid_chunk.shape)

        # Evaluate the log-likelihood for each point in the chunk
        for i in range(A_grid_chunk.shape[0]):
            for j in range(A_grid_chunk.shape[1]):
                for k in range(A_grid_chunk.shape[2]):
                    log_likelihood_values_chunk[i, j, k] = likelihood(
                        A=A_grid_chunk[i, j, k],
                        v_0=v_0_grid_chunk[i, j, k],
                        alpha=alpha_grid_chunk[i, j, k],
                        v_i=nu,
                        s_n=s_n,
                    )

        return log_likelihood_values_chunk

    def compute_pdfs(
        self,
        model: Callable[..., Any],
        A: np.ndarray,
        v0: np.ndarray,
        alpha: np.ndarray,
        num_chunks: int = 10,
    ) -> tuple:
        """
        Compute the marginalized PDFs for parameters A, v0, and alpha based on the
        model and likelihood function. The grid is divided into chunks and processed
        in parallel to speed up computation.

        :param model: The generative model or likelihood function to evaluate.
        :type model: Callable[..., Any]
        :param A: The grid of parameter A values.
        :type A: np.ndarray
        :param v0: The grid of parameter v0 values.
        :type v0: np.ndarray
        :param alpha: The grid of parameter alpha values.
        :type alpha: np.ndarray
        :param num_chunks: The number of chunks to split the grid
        for parallel processing (default is 10).
        :type num_chunks: int, optional
        :return: A tuple containing the marginalized PDFs for various parameter combinations:
                 (P(A, ν0), P(A, α), P(ν0, α), P(A), P(ν0), P(α)).
        :rtype: tuple
        """
        # Create the meshgrid
        A_grid, v_0_grid, alpha_grid = np.meshgrid(A, v0, alpha, indexing='ij')

        # Initialize an array to store the log-likelihood values
        log_likelihood_values = np.zeros(A_grid.shape)

        # Number of chunks to split the work into
        # (based on the number of available CPU cores)
        chunk_size = A_grid.shape[0] // num_chunks

        # Split the grid into chunks for parallel processing
        chunks = []
        for i in range(num_chunks):
            chunk_A_grid = A_grid[i * chunk_size : (i + 1) * chunk_size, :, :]
            chunk_v_0_grid = v_0_grid[
                i * chunk_size : (i + 1) * chunk_size, :, :
            ]
            chunk_alpha_grid = alpha_grid[
                i * chunk_size : (i + 1) * chunk_size, :, :
            ]
            chunks.append(
                (
                    chunk_A_grid,
                    chunk_v_0_grid,
                    chunk_alpha_grid,
                    model,
                    self._nu,
                    self._s_n,
                )
            )

        # Create a pool of worker processes to compute
        # the log-likelihood in parallel
        with Pool(processes=num_chunks) as pool:
            log_likelihood_chunks = pool.map(
                self._compute_log_likelihood_chunk, chunks
            )

        # Combine the chunks into the final
        # log-likelihood values array
        log_likelihood_values = np.concatenate(log_likelihood_chunks, axis=0)

        # Get the likelihood
        likelihood_values = np.exp(log_likelihood_values)

        # Normalize the likelihood values by deviding by their sum
        normalized_likelihood_values = likelihood_values / np.sum(
            likelihood_values
        )

        # Extract all marginalized pdf needed
        marginalized_A_v0 = np.sum(normalized_likelihood_values, axis=2)
        marginalized_A_alpha = np.sum(normalized_likelihood_values, axis=1)
        marginalized_v0_alpha = np.sum(normalized_likelihood_values, axis=0)
        marginalized_A = np.sum(marginalized_A_v0, axis=1)
        marginalized_v0 = np.sum(marginalized_v0_alpha, axis=1)
        marginalized_alpha = np.sum(marginalized_v0_alpha, axis=0)

        return (
            marginalized_A_v0,  # P(A,νo)
            marginalized_A_alpha,  # P(A,α)
            marginalized_v0_alpha,  # P(νo,α)
            marginalized_A,  # P(A)
            marginalized_v0,  # P(νo)
            marginalized_alpha,  # P(α)
        )
