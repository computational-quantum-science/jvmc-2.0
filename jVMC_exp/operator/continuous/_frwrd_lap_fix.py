import fwdlap
import jax
import jax.numpy as jnp
from jax._src.ad_util import Zero, SymbolicZero

def _patched_zero_tangent_from_primal(primal, jsize=None):
    aval = jax.core.get_aval(primal)
    if hasattr(aval, 'to_tangent_aval'):
        aval = aval.to_tangent_aval()
    zero = Zero(aval)
    if jsize is None:
        return zero
    aval = zero.aval
    return Zero(aval.update(shape=(jsize, *aval.shape)))

fwdlap.zero_tangent_from_primal = _patched_zero_tangent_from_primal

def _instantiate_symbolic_zeros(xs, primals):
    result = []
    for t, p in zip(xs, primals):
        if type(t) is Zero or isinstance(t, SymbolicZero):
            result.append(jnp.zeros_like(p))
        else:
            result.append(t)

    return result

def _patched_vhv_by_jvp(f_jvp, jsize, primals_in, jacs_in, laps_in, **params):
    laps_in = _instantiate_symbolic_zeros(laps_in, primals_in)
    jacs_in = _instantiate_symbolic_zeros(jacs_in, primals_in)

    return fwdlap.vhv_by_jvp(f_jvp, jsize, primals_in, jacs_in, laps_in, **params)

fwdlap.vhv_by_jvp = _patched_vhv_by_jvp

from fwdlap import lap