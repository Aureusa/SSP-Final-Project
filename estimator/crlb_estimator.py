import sympy as sp
import numpy as np


class CRLBEstimator:
    """
    A class to compute the Cramér-Rao Lower Bound (CRLB)
    and compute the Fisher Information Matrix.
    """

    def compute_crlb(self, fisher_matrix: np.ndarray):
        """
        Computes the Cramér-Rao Lower Bound (CRLB) using
        the Fisher Information Matrix.

        :param fisher_matrix: The Fisher Information Matrix.
        :type fisher_matrix: np.ndarray
        :return: The CRLB for each parameter as a 1D array.
        :rtype: np.ndarray
        """
        # Takes the inverse
        inv_fisher_matrix = np.linalg.inv(fisher_matrix)

        # Gets the diagonal elemnts of the inverse fisher matrix
        crlb = np.array(
            [
                inv_fisher_matrix[0][0],
                inv_fisher_matrix[1][1],
                inv_fisher_matrix[2][2],
            ]
        )

        return crlb

    def compute_fisher_matrix(
        self, params: tuple, data: np.ndarray
    ) -> np.ndarray:
        """
        Computes the Fisher Information Matrix for a generative
        model based on given parameters and data.

        :param params: A tuple containing model parameters (A, v_0, alpha).
        :type params: tuple
        :param data: The independent variable values v_i.
        :type data: np.ndarray
        :return: The Fisher Information Matrix.
        :rtype: np.ndarray
        """
        model, A, v_0, alpha, v_i = self._get_model_with_symbols()

        df_dA, df_dv_0, df_dalpha = self._find_derivatives(
            model, A, v_0, alpha
        )

        # Evaluate derivatives
        df_dA_func = sp.lambdify((A, v_0, alpha, v_i), df_dA, 'numpy')
        df_dv_0_func = sp.lambdify((A, v_0, alpha, v_i), df_dv_0, 'numpy')
        df_dalpha_func = sp.lambdify((A, v_0, alpha, v_i), df_dalpha, 'numpy')

        # Unpack the parameters
        A, v_0, alpha = params

        v_i = data

        # Evaluate the derivatives
        df_dA_eval = df_dA_func(A, v_0, alpha, v_i)
        df_dv_0_eval = df_dv_0_func(A, v_0, alpha, v_i)
        df_dalpha_eval = df_dalpha_func(A, v_0, alpha, v_i)

        # Compute the different rows of the Fisher Matrix
        # First row
        i_1_1 = np.mean(df_dA_eval * df_dA_eval)
        i_1_2 = np.mean(df_dA_eval * df_dv_0_eval)
        i_1_3 = np.mean(df_dA_eval * df_dalpha_eval)

        # Second row
        i_2_1 = np.mean(df_dv_0_eval * df_dA_eval)
        i_2_2 = np.mean(df_dv_0_eval * df_dv_0_eval)
        i_2_3 = np.mean(df_dv_0_eval * df_dalpha_eval)

        # Third row
        i_3_1 = np.mean(df_dalpha_eval * df_dA_eval)
        i_3_2 = np.mean(df_dalpha_eval * df_dv_0_eval)
        i_3_3 = np.mean(df_dalpha_eval * df_dalpha_eval)

        fisher_matrix = (len(v_i) / 0.05**2) * np.array(
            [
                [i_1_1, i_1_2, i_1_3],
                [i_2_1, i_2_2, i_2_3],
                [i_3_1, i_3_2, i_3_3],
            ]
        )
        return fisher_matrix

    def _find_derivatives(
        self,
        generative_model: sp.Expr,
        A: sp.Symbol,
        v_0: sp.Symbol,
        alpha: sp.Symbol,
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        """
        Computes the partial derivatives of the generative
        model with respect to parameters.

        :param generative_model: The symbolic representation
        of the generative model.
        :type generative_model: sp.Expr
        :param A: The symbol representing parameter A.
        :type A: sp.Symbol
        :param v_0: The symbol representing parameter v_0.
        :type v_0: sp.Symbol
        :param alpha: The symbol representing parameter alpha.
        :type alpha: sp.Symbol
        :return: A tuple of partial derivatives with respect to A, v_0, and alpha.
        :rtype: tuple[sp.Expr, sp.Expr, sp.Expr]
        """
        df_dA = sp.diff(generative_model, A)
        df_dv_0 = sp.diff(generative_model, v_0)
        df_dalpha = sp.diff(generative_model, alpha)
        return df_dA, df_dv_0, df_dalpha

    def _get_model_with_symbols(
        self,
    ) -> tuple[sp.Expr, sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
        """
        Defines the generative model and its
        parameters as symbolic expressions.

        :return: The generative model and its
        symbolic parameters (A, v_0, alpha, v_i).
        :rtype: tuple[sp.Expr,sp.Symbol,sp.Symbol,sp.Symbol,sp.Symbol]
        """
        A, v_0, alpha, v_i = sp.symbols(['A', 'v_0', 'alpha', 'v_i'])

        model = A * (v_i / v_0) ** alpha * (1 + v_i / v_0) ** (-4 * alpha)
        return model, A, v_0, alpha, v_i
