import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

DT_SAMPLES_CONT = jnp.float64
DT_SAMPLES = jnp.int32
DT_OPERATORS_REAL = jnp.float64
DT_OPERATORS_CPX = jnp.complex128
DT_PARAMS_REAL = jnp.float64
DT_PARAMS_CPX = jnp.complex128