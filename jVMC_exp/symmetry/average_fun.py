import jax.numpy as jnp
from typing import Literal, Callable

def avg_coefficients_exp(log_amplitudes, weights):
    log_amplitudes = jnp.asarray(log_amplitudes, dtype=jnp.complex128)
    weights = jnp.asarray(weights, dtype=jnp.complex128)
    shift = jnp.max(jnp.real(log_amplitudes))

    return shift + jnp.log(jnp.mean(weights * jnp.exp(log_amplitudes - shift)))

def avg_coefficients_log(log_amplitudes, weights):
    del weights
    return jnp.mean(jnp.asarray(log_amplitudes, dtype=jnp.complex128))

def avg_coefficients_sep(log_amplitudes, weights):
    del weights
    log_amplitudes = jnp.asarray(log_amplitudes, dtype=jnp.complex128)
    re = jnp.real(log_amplitudes)
    im = jnp.imag(log_amplitudes)

    return 0.5 * jnp.log(jnp.mean(jnp.exp(2 * re))) + 1j * jnp.angle(jnp.mean(jnp.exp(1j * im)))

average_coefficients_table = {
    "exp": avg_coefficients_exp,
    "log": avg_coefficients_log,
    "sep": avg_coefficients_sep
}
BuiltinSymmetryAverage = Literal["exp", "log", "sep"]
SymmetryAverage = BuiltinSymmetryAverage | Callable

def average_coefficients(log_amplitudes, weights, average: SymmetryAverage):
    if callable(average):
        return jnp.asarray(
            average(log_amplitudes, weights),
            dtype=jnp.complex128,
        )

    fn = average_coefficients_table.get(average)
    if fn is None:
        allowed = ", ".join(k for k in average_coefficients_table.keys())
        raise ValueError(
            f"symmetry_average must be one of {allowed}, or a callable."
        )

    return fn(log_amplitudes, weights)