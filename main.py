from estimator.estimator import MLEGeneralJAX
from model.model_constructor import ModelConstructor
from utils import get_data
import jax.numpy as jnp

freq, signal, signal_noise = get_data()
nu = jnp.array(freq.tolist())
s_n = jnp.array(signal_noise.tolist())
DATA = tuple((nu, s_n))

INITIAL_GUESS = [6.0, 2.0, 1.0]


def mle_general_jax(initial_guess: list[float]):
    model_constructor = ModelConstructor()

    log_likelihood = model_constructor.get_log_likelihood()
    generative_model = model_constructor.get_generative_model()

    nuisance_params = ["v", "s_n", "sigma"]

    mle_estimator = MLEGeneralJAX(
        model=generative_model,
        log_likelihood=log_likelihood,
        nuisance_params=nuisance_params,
        data=DATA,
        gamma=0.1,
    )

    guess = mle_estimator.evaluate_the_ML_estimate(
        guess=initial_guess, data_logging=True, treshhold=1e-6
    )

    mle_estimator.plot_gradient_norms()
    mle_estimator.plot_estimates()
    mle_estimator.plot_residuals(guess)
    # print(guess)


if __name__ == "__main__":
    mle_general_jax(INITIAL_GUESS)
